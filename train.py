import os
# Restrict thread contention between underlying math libraries and OpenCV to prevent excessive CPU thread overhead
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_UPDATE_CHECK"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['ALBUMENTATIONS_SUPPRESS_CHECK'] = '1'

import sys
import random
import argparse
import math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F

# Disable OpenCV internal multithreading to avoid conflicts with DataLoader multiprocessing
cv2.setNumThreads(0)

from model import DualSwinV2MonaUperNet


class Tee:
    def __init__(self, filename, mode='w', encoding='utf-8'):
        self.stdout = sys.stdout
        self.file = open(filename, mode, encoding=encoding)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    sys.stdout = Tee(log_filename)
    print(f"Logging output to: {log_filename}")


def clear_checkpoints_directory(dir_path='model_weights'):
    if not os.path.exists(dir_path):
        return
    files_to_delete = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    for file_name in files_to_delete:
        try:
            os.remove(os.path.join(dir_path, file_name))
        except OSError:
            pass


def replicate_pil_convert_rgb(image_array: np.ndarray) -> np.ndarray:
    if image_array.ndim == 2:
        return np.stack([image_array] * 3, axis=-1)
    elif image_array.ndim == 3:
        num_channels = image_array.shape[2]
        if num_channels == 1:
            return image_array.repeat(3, axis=2)
        elif num_channels >= 3:
            return image_array[:, :, :3]
    raise ValueError(f"Unsupported image dimensions: {image_array.shape}")


class MetricsCalculator:
    def __init__(self, num_classes, ignore_index, num_eval_classes=None):
        self.num_total_classes = num_classes
        self.ignore_index = ignore_index
        self.num_eval_classes = num_eval_classes if num_eval_classes is not None else num_classes
        self.confusion_matrix = np.zeros((self.num_total_classes, self.num_total_classes), dtype=np.int64)

    def update(self, preds, labels):
        preds = preds.cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        mask = (labels >= 0) & (labels < self.num_total_classes)
        labels_masked, preds_masked = labels[mask], preds[mask]

        if labels_masked.size > 0:
            preds_masked[preds_masked >= self.num_total_classes] = self.num_total_classes - 1
            cm = np.bincount(
                self.num_total_classes * labels_masked.astype(np.int64) + preds_masked.astype(np.int64),
                minlength=self.num_total_classes ** 2
            ).reshape(self.num_total_classes, self.num_total_classes)
            self.confusion_matrix += cm

    def compute(self, class_names):
        if self.confusion_matrix.sum() == 0:
            oa = 0
            iou_all_classes = np.zeros(self.num_total_classes)
            accuracy_all_classes = np.zeros(self.num_total_classes)
            f1_score_all_classes = np.zeros(self.num_total_classes)
        else:
            full_tp_sum = np.diag(self.confusion_matrix).sum()
            full_pixel_sum = self.confusion_matrix.sum()
            oa = full_tp_sum / full_pixel_sum if full_pixel_sum > 0 else 0

            tp = np.diag(self.confusion_matrix)
            fp = self.confusion_matrix.sum(axis=0) - tp
            fn = self.confusion_matrix.sum(axis=1) - tp

            with np.errstate(divide='ignore', invalid='ignore'):
                iou_all_classes = tp / (tp + fp + fn)
                accuracy_all_classes = tp / self.confusion_matrix.sum(axis=1)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score_all_classes = 2 * (precision * recall) / (precision + recall)

            iou_all_classes[np.isnan(iou_all_classes)] = 0
            accuracy_all_classes[np.isnan(accuracy_all_classes)] = 0
            f1_score_all_classes[np.isnan(f1_score_all_classes)] = 0

        iou_eval_classes = iou_all_classes[:self.num_eval_classes]
        f1_eval_classes = f1_score_all_classes[:self.num_eval_classes]
        miou = np.nanmean(iou_eval_classes)
        mf1 = np.nanmean(f1_eval_classes)

        header = f"{'Class':<25} | {'IoU':>6} | {'Accuracy':>10} | {'F1-Score':>10}"
        separator = "-" * len(header)

        result_str_list = [
            f"Overall Accuracy (OA): {oa:.4f}",
            f"Mean IoU (mIoU):       {miou:.4f}",
            f"Mean F1 (mF1):         {mf1:.4f}\n",
            header,
            separator]

        for i, name in enumerate(class_names):
            result_str_list.append(
                f"{name:<25} | {iou_all_classes[i]:>6.4f} | {accuracy_all_classes[i]:>10.4f} | {f1_score_all_classes[i]:>10.4f}")

        result_str_list.append(separator)
        metrics_dict = {'OA': oa, 'mIoU': miou, 'mF1': mf1,
                        'per_class_iou': iou_eval_classes,
                        'per_class_f1': f1_eval_classes}
        return "\n".join(result_str_list), metrics_dict


def get_file_paths(config, areas_list, purpose='train'):
    file_paths = []
    templates = config['ACTIVE_DATASET']['FILE_TEMPLATES']
    base_dir = config['ACTIVE_DATASET']['DATA_DIR']
    gt_template_key = 'gt'
    if purpose == 'val' and 'gt_val' in templates:
        gt_template_key = 'gt_val'
    for area in areas_list:
        top_path = os.path.join(base_dir, templates['top'].format(area))
        dsm_path = os.path.join(base_dir, templates['dsm'].format(area))
        gt_path = os.path.join(base_dir, templates[gt_template_key].format(area))
        if os.path.exists(top_path) and os.path.exists(dsm_path) and os.path.exists(gt_path):
            file_paths.append({'top': top_path, 'dsm': dsm_path, 'gt': gt_path})
    if not file_paths:
        raise RuntimeError(f"No image files found at path: {base_dir}")
    return file_paths


def setup_data_preprocessing(config, is_train=True):
    aug_config = config['AUGMENTATION']
    img_size = config['DATA']['IMG_SIZE']
    if is_train:
        return A.Compose([
            A.RandomRotate90(p=aug_config['geometry_prob']),
            A.VerticalFlip(p=aug_config['geometry_prob']),
            A.HorizontalFlip(p=aug_config['geometry_prob']),
            A.Compose([
                A.ColorJitter(brightness=aug_config['brightness'], contrast=aug_config['contrast'],
                              saturation=aug_config['saturation'], hue=aug_config['hue'],
                              p=aug_config['color_jitter_prob']),
            ]),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], additional_targets={'dsm': 'image'})
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def _create_palette_map(data_config, num_classes, ignore_index):
    mapping = {tuple(v): k for k, v in data_config['PALETTE'].items() if k < num_classes}
    mapping[(0, 0, 0)] = ignore_index
    return mapping


def _convert_gt_to_labels(gt_rgb, palette_map, ignore_index):
    h, w, _ = gt_rgb.shape
    gt_labels = np.full((h, w), ignore_index, dtype=np.uint8)
    for rgb, class_idx in palette_map.items():
        mask = np.all(gt_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        gt_labels[mask] = class_idx
    return gt_labels


class DynamicTrainDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_config = config['ACTIVE_DATASET']
        self.crop_config = config['DATA']['MULTI_SCALE_TRAIN_CROP']
        self.augmentations = setup_data_preprocessing(config, is_train=True)
        self.palette_map = _create_palette_map(self.data_config, config['MODEL']['NUM_CLASSES'],
                                               config['DATA']['IGNORE_INDEX'])
        self.ignore_index = config['DATA']['IGNORE_INDEX']
        self.scales = self.crop_config['scales']
        self.total_count = self.crop_config['total_count']

        try:
            self.car_class_index = self.data_config['CLASSES'].index('Cars')
        except ValueError:
            self.car_class_index = -1

        print("Preloading training data into RAM, please wait...")
        raw_file_paths = get_file_paths(config, self.data_config['TRAIN_AREAS'], purpose='train')
        self.cached_data = []
        for path_dict in tqdm(raw_file_paths, desc="Loading TIFs to RAM"):
            top_img = replicate_pil_convert_rgb(tifffile.imread(path_dict['top']))
            dsm_img = tifffile.imread(path_dict['dsm'])
            gt_img = replicate_pil_convert_rgb(tifffile.imread(path_dict['gt']))
            gt_labels = _convert_gt_to_labels(gt_img, self.palette_map, self.ignore_index)
            self.cached_data.append({
                'top': top_img,
                'dsm': dsm_img,
                'gt': gt_labels
            })
        print(f"Data preloading complete, {len(self.cached_data)} regions loaded.")

    def __len__(self):
        return self.total_count

    def __getitem__(self, idx):
        spin_count = 0
        while True:
            spin_count += 1
            try:
                data_item = random.choice(self.cached_data)
                scale_config = random.choices(self.scales, weights=[s['proportion'] for s in self.scales], k=1)[0]
                size = scale_config['size']
                h, w = data_item['top'].shape[0], data_item['top'].shape[1]

                if h < size or w < size:
                    continue

                x, y = random.randint(0, w - size), random.randint(0, h - size)

                top_crop = data_item['top'][y:y + size, x:x + size]
                dsm_crop = data_item['dsm'][y:y + size, x:x + size]
                gt_crop_labels = data_item['gt'][y:y + size, x:x + size]

                dsm_crop = np.expand_dims(dsm_crop, axis=-1)

                # Infinite loop guard: proceed if no cars found after 30 retries to prevent CPU spinlock
                if spin_count <= 30 and self.car_class_index != -1 and self.car_class_index not in gt_crop_labels:
                    if random.random() < 0.5:
                        continue

                gt_labels_expanded = np.expand_dims(gt_crop_labels, axis=-1)
                transformed = self.augmentations(image=top_crop, dsm=dsm_crop, masks=[gt_labels_expanded])

                top_tensor = transformed['image']
                dsm_tensor = transformed['dsm'].float() / 255.0
                gt_tensor = transformed['masks'][0].squeeze(0).long()

                return top_tensor, dsm_tensor, gt_tensor

            except Exception:
                continue


class InMemoryDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def setup_validation_preprocessing():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'dsm': 'image'})


def create_validation_dataset(config):
    import gc
    print("\n--- Generating fixed validation dataset ---")
    data_config = config['ACTIVE_DATASET']
    val_file_paths = get_file_paths(config, data_config['TEST_AREAS'], purpose='val')
    val_transforms = setup_validation_preprocessing()
    palette_map = _create_palette_map(data_config, config['MODEL']['NUM_CLASSES'], config['DATA']['IGNORE_INDEX'])
    ignore_index = config['DATA']['IGNORE_INDEX']
    val_crop_count = config['DATA']['VAL_CROP_COUNT']
    img_size = config['DATA']['IMG_SIZE']

    temp_val_cache = []
    for path_dict in val_file_paths:
        top_img = replicate_pil_convert_rgb(tifffile.imread(path_dict['top']))
        dsm_img = tifffile.imread(path_dict['dsm'])
        gt_img = replicate_pil_convert_rgb(tifffile.imread(path_dict['gt']))
        gt_labels = _convert_gt_to_labels(gt_img, palette_map, ignore_index)
        temp_val_cache.append({'top': top_img, 'dsm': dsm_img, 'gt': gt_labels})

    val_samples = []
    pbar_val = tqdm(range(val_crop_count), desc="Generating validation samples")

    while len(val_samples) < val_crop_count:
        data_item = random.choice(temp_val_cache)
        h, w = data_item['top'].shape[:2]
        if h < img_size or w < img_size:
            continue
        x, y = random.randint(0, w - img_size), random.randint(0, h - img_size)

        top_crop = data_item['top'][y:y + img_size, x:x + img_size]
        dsm_crop = np.expand_dims(data_item['dsm'][y:y + img_size, x:x + img_size], axis=-1)
        gt_crop_labels = np.expand_dims(data_item['gt'][y:y + img_size, x:x + img_size], axis=-1)

        transformed = val_transforms(image=top_crop, dsm=dsm_crop, masks=[gt_crop_labels])
        top_tensor = transformed['image']
        dsm_tensor = transformed['dsm'].float() / 255.0
        gt_tensor = transformed['masks'][0].squeeze(0).long()

        val_samples.append((top_tensor, dsm_tensor, gt_tensor))
        pbar_val.update(1)

    pbar_val.close()
    del temp_val_cache
    gc.collect()
    return InMemoryDataset(val_samples)


def load_pretrained_weights(model, weights_path):
    if not os.path.exists(weights_path):
        print(f"⚠️ Pretrained weights not found: '{weights_path}'. Training from scratch.")
        return model

    checkpoint = torch.load(weights_path, map_location='cpu')
    pretrained_dict = checkpoint.get('model', checkpoint)
    model_dict = model.state_dict()
    weights_to_load = {}

    for k_pre, v_pre in pretrained_dict.items():
        if any(n in k_pre for n in ["relative_position_index", "attn_mask", "head", "relative_coords_table"]):
            continue

        k_x_base, k_y_base = None, None
        if k_pre.startswith('patch_embed.'):
            k_x_base, k_y_base = k_pre.replace('patch_embed.', 'patch_embed_x.'), k_pre.replace('patch_embed.', 'patch_embed_y.')
        elif k_pre.startswith('layers.'):
            k_x_base, k_y_base = k_pre.replace('layers.', 'layers_x.'), k_pre.replace('layers.', 'layers_y.')

        for k_target_base in [k_x_base, k_y_base]:
            if not k_target_base:
                continue

            lora_keywords = ['attn.qkv.weight', 'attn.proj.weight', 'attn.proj.bias',
                             'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']
            is_lora = False
            for keyword in lora_keywords:
                if k_target_base.endswith(keyword):
                    k_target_lora = k_target_base.replace(keyword.split('.')[-1], f'original_layer.{keyword.split(".")[-1]}')
                    if k_target_lora in model_dict and v_pre.shape == model_dict[k_target_lora].shape:
                        weights_to_load[k_target_lora] = v_pre
                        is_lora = True
                        break
            if not is_lora:
                if k_target_base in model_dict and v_pre.shape == model_dict[k_target_base].shape:
                    weights_to_load[k_target_base] = v_pre

    model_dict.update(weights_to_load)
    model.load_state_dict(model_dict, strict=False)
    print(f"✅ Successfully loaded {len(weights_to_load)} pretrained weight tensors")
    return model


def set_parameter_requires_grad(model, config):
    lora_config = config['MODEL']['SWINV2'].get('LORA', {})
    lora_rank = lora_config.get('rank', 0)

    if lora_rank <= 0:
        for layer_name in ['layers_x', 'layers_y']:
            for layer in getattr(model, layer_name):
                for block in layer.blocks:
                    for param in block.attn.parameters():
                        param.requires_grad = False
                    for param in block.mlp.parameters():
                        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f} M | Trainable params: {trainable_params / 1e6:.2f} M")
    return model


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index, class_weights, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, targets):
        # Force cast to FP32 for probability calculation and summation to prevent denominator overflow to inf above 65504
        logits_fp32 = logits.float()
        probs = F.softmax(logits_fp32, dim=1)

        mask = (targets != self.ignore_index)
        targets_for_onehot = targets.clone()
        targets_for_onehot[~mask] = 0
        targets_one_hot = F.one_hot(targets_for_onehot, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        mask_expanded = mask.unsqueeze(1).expand_as(probs)
        probs_masked = probs * mask_expanded
        targets_masked = targets_one_hot * mask_expanded

        # Accumulation is performed in FP32 precision, supporting a dynamic range up to 3.4e38
        intersection = torch.sum(probs_masked * targets_masked, dim=(0, 2, 3))
        cardinality = torch.sum(probs_masked, dim=(0, 2, 3)) + torch.sum(targets_masked, dim=(0, 2, 3))
        dice_score_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        weighted_dice_score = dice_score_per_class * self.class_weights
        sum_of_weights = torch.sum(self.class_weights)

        if sum_of_weights > 0:
            return 1.0 - (torch.sum(weighted_dice_score) / sum_of_weights)

        return torch.tensor(0.0, device=logits.device)


class CombinedLoss(nn.Module):
    def __init__(self, alpha, beta, class_weights_ce, class_weights_dice, num_classes, ignore_index):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights_ce, ignore_index=ignore_index)
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index, class_weights=class_weights_dice)

    def forward(self, logits, targets):
        return self.alpha * self.ce_loss(logits, targets) + self.beta * self.dice_loss(logits, targets)


def main(config):
    original_stdout = sys.stdout
    try:
        clear_checkpoints_directory(dir_path='model_weights')
        setup_logging()

        print(f"--- Starting Training --- \nDataset: {config['ACTIVE_DATASET']['DATASET_NAME']}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        print(f"Using device: {device}")

        print("\n" + "#" * 50)
        if torch.cuda.is_available():
            current_device_id = torch.cuda.current_device()
            print(f"🔥 Current CUDA Device ID: {current_device_id}")
            print(f"🔥 Device Name: {torch.cuda.get_device_name(current_device_id)}")
            print(f"🔥 Total Physical VRAM: {torch.cuda.get_device_properties(current_device_id).total_memory / (1024**3):.2f} GB")
        else:
            print("⚠️ Warning: No CUDA device detected. Running on CPU!")
        print("#" * 50 + "\n")

        num_workers = config['DATA']['NUM_WORKERS']
        print(f"🚀 Using {num_workers} worker processes for data loading.")
        dl_kwargs = {'batch_size': config['TRAIN']['BATCH_SIZE'], 'num_workers': num_workers, 'pin_memory': True}
        if num_workers > 0:
            dl_kwargs['persistent_workers'] = True

        val_dataset = create_validation_dataset(config)
        val_loader = DataLoader(val_dataset, **dl_kwargs)

        train_dataset = DynamicTrainDataset(config)
        train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)

        model = DualSwinV2MonaUperNet(config)
        model = load_pretrained_weights(model, config['TRAIN']['PRETRAINED_WEIGHTS_PATH'])
        model = set_parameter_requires_grad(model, config)
        model.to(device)

        loss_config = config['TRAIN']['LOSS']
        class_weights_ce = torch.tensor(config['ACTIVE_DATASET']['CLASS_WEIGHTS_CE'], dtype=torch.float)
        criterion = CombinedLoss(
            alpha=loss_config['alpha'],
            beta=loss_config['beta'],
            class_weights_ce=class_weights_ce,
            class_weights_dice=loss_config['CLASS_WEIGHTS_DICE'],
            num_classes=config['MODEL']['NUM_CLASSES'],
            ignore_index=config['DATA']['IGNORE_INDEX']
        ).to(device)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=config['TRAIN']['OPTIMIZER']['lr'],
                                      weight_decay=config['TRAIN']['OPTIMIZER']['weight_decay'])
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        grad_clip_norm = config['TRAIN']['GRAD_CLIP_MAX_NORM']
        grad_accum_steps = config['TRAIN']['GRAD_ACCUMULATION_STEPS']

        base_lr = config['TRAIN']['OPTIMIZER']['lr']
        min_lr = base_lr * config['TRAIN']['LR_SCHEDULER']['min_lr_ratio']
        warmup_epochs = config['TRAIN']['LR_SCHEDULER']['warmup_epochs']
        total_epochs = config['TRAIN']['EPOCHS']

        best_metric, patience_counter = 0.0, 0
        num_eval_classes = config['ACTIVE_DATASET']['NUM_EVAL_CLASSES']
        effective_class_names = config['ACTIVE_DATASET']['CLASSES'][:num_eval_classes]

        for epoch in range(total_epochs):
            print(f"\n===== Epoch {epoch + 1}/{total_epochs} =====")

            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")
            running_loss_tensor = torch.zeros(1, device=device)
            optimizer.zero_grad(set_to_none=True)

            for i, (top, dsm, gt) in enumerate(train_pbar):
                top = top.to(device, non_blocking=True)
                dsm = dsm.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    if dsm.ndim == 4 and dsm.shape[3] == 1:
                        dsm = dsm.permute(0, 3, 1, 2).contiguous()
                    if gt.ndim == 4 and gt.shape[3] == 1:
                        gt = gt.squeeze(3)
                    outputs = model(top, dsm)
                    loss = criterion(outputs, gt)

                loss_scaled = loss / grad_accum_steps
                scaler.scale(loss_scaled).backward()

                if (i + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss_tensor += loss.detach()
                if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                    current_avg_loss = running_loss_tensor.item() / (i + 1)
                    train_pbar.set_postfix(loss=f"{current_avg_loss:.4f}", lr=f"{lr:.2e}")

            epoch_avg_loss = running_loss_tensor.item() / len(train_loader)
            print(f"\n[Epoch {epoch + 1} End] Average Training Loss: {epoch_avg_loss:.4f}")

            # Validation phase
            model.eval()
            metrics_calculator = MetricsCalculator(
                num_classes=config['MODEL']['NUM_CLASSES'],
                ignore_index=config['DATA']['IGNORE_INDEX'],
                num_eval_classes=config['ACTIVE_DATASET'].get('NUM_EVAL_CLASSES')
            )
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation")
            with torch.no_grad():
                for top, dsm, gt in val_pbar:
                    top = top.to(device, non_blocking=True)
                    dsm = dsm.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                        if dsm.ndim == 4 and dsm.shape[3] == 1:
                            dsm = dsm.permute(0, 3, 1, 2).contiguous()
                        if gt.ndim == 4 and gt.shape[3] == 1:
                            gt = gt.squeeze(3)
                        outputs = model(top, dsm)
                    preds = torch.argmax(outputs, dim=1)
                    metrics_calculator.update(preds, gt)

            result_str, metrics_dict = metrics_calculator.compute(effective_class_names)
            print(result_str)
            current_metric = metrics_dict.get('mIoU', 0)

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                model_weights_dir = "model_weights"
                os.makedirs(model_weights_dir, exist_ok=True)
                for f in os.listdir(model_weights_dir):
                    if f.endswith('.pth'):
                        try:
                            os.remove(os.path.join(model_weights_dir, f))
                        except OSError:
                            pass
                torch.save(model.state_dict(), os.path.join(model_weights_dir, f"mIoU_{current_metric:.4f}.pth"))
            else:
                patience_counter += 1

            if patience_counter >= config['TRAIN']['EARLY_STOPPING']['patience']:
                print("Early stopping triggered. Training terminated.")
                break

        print("\n--- Training Complete ---")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if isinstance(sys.stdout, Tee):
            sys.stdout.close()
        sys.stdout = original_stdout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dual-stream semantic segmentation model with dynamic data generation.")
    parser.add_argument('--dataset', type=str, default='vaihingen', choices=['potsdam', 'vaihingen'])
    args = parser.parse_args()

    if args.dataset.lower() == 'potsdam':
        from config.configs_potsdam import get_config
    else:
        from config.configs_vaihingen import get_config

    config = get_config(args.dataset)

    import warnings
    warnings.filterwarnings("ignore")

    main(config)
