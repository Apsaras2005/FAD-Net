_VAIHINGEN_CONFIG_BASE = {
    'DATASET_NAME': 'Vaihingen',
    'DATA_DIR': 'datasets/Vaihingen',
    'CLASSES': ['Impervious surfaces', 'Buildings', 'Low vegetation', 'Trees', 'Cars', 'Clutter'],
    'NUM_EVAL_CLASSES': 5,
    'PALETTE': {
        0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255), 3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0),
    },
    'CLASS_WEIGHTS_CE': [1.0, 0.7, 2.0, 1.2, 3.0, 0.8],
    'TRAIN_AREAS': [
        '1', '3', '5', '7', '11', '13', '15', '17', '21', '23', '26', '28', '30', '32', '34', '37'
    ],
    'TEST_AREAS': [
        '2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38'
    ],
    'FILE_TEMPLATES': {
        'top': 'top/top_mosaic_09cm_area{}.tif',
        'dsm': 'dsm/dsm_09cm_matching_area{}.tif',
        'gt': 'ground_truth/top_mosaic_09cm_area{}.tif',
        'gt_val': 'ground_truth/top_mosaic_09cm_area{}_noBoundary.tif'
    }
}


_POTSDAM_CONFIG_BASE = {
    'DATASET_NAME': 'Potsdam',
    'DATA_DIR': 'datasets/Potsdam',
    'CLASSES': ['Impervious surfaces', 'Buildings', 'Low vegetation', 'Trees', 'Cars', 'Clutter'],
    'NUM_EVAL_CLASSES': 5,
    'PALETTE': {
        0: (255, 255, 255), 1: (0, 0, 255), 2: (0, 255, 255), 3: (0, 255, 0), 4: (255, 255, 0), 5: (255, 0, 0),
    },
    'CLASS_WEIGHTS_CE': [1.0, 0.7, 1.4, 1.2, 1.7, 1.3],
    'TRAIN_AREAS': [
        '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12', '5_10', '5_11', '5_12',
        '6_07', '6_08', '6_09', '6_10', '6_11', '6_12', '7_07', '7_08', '7_09', '7_10', '7_11', '7_12'
    ],
    'TEST_AREAS': [
        '2_13', '2_14', '3_13', '3_14', '4_13', '4_14', '4_15', '5_13', '5_14', '5_15', '6_13', '6_14', '6_15', '7_13'
    ],
    'FILE_TEMPLATES': {
        'top': 'top/top_potsdam_{}_RGB.tif',
        'dsm': 'dsm/dsm_potsdam_{}_normalized_lastools.tif',
        'gt': 'ground_truth/top_potsdam_{}_label.tif',
        'gt_val': 'ground_truth/top_potsdam_{}_label_noBoundary.tif'
    }
}

def get_config(dataset_name='potsdam'):
    if dataset_name.lower() == 'potsdam':
        active_dataset = _POTSDAM_CONFIG_BASE
    elif dataset_name.lower() == 'vaihingen':
        active_dataset = _VAIHINGEN_CONFIG_BASE
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    CONFIG = {
        'ACTIVE_DATASET': active_dataset,
        'DATA': {
            'IMG_SIZE': 256,
            'NUM_WORKERS': 8,
            'MULTI_SCALE_TRAIN_CROP': {
                'total_count': 2000,
                'scales': [
                    {'size': 1024, 'proportion': 0.2}, {'size': 512, 'proportion': 0.2},
                    {'size': 256, 'proportion': 0.4}, {'size': 128, 'proportion': 0.2},
                ]
            },
            'VAL_CROP_COUNT': 2000,
            'IGNORE_INDEX': 255
        },
        'AUGMENTATION': {
            'geometry_prob': 0.75, 'color_jitter_prob': 0.5, 'brightness': 0.25,
            'contrast': 0.25, 'saturation': 0.2, 'hue': 0.1
        },
        'TRAIN': {
            'EPOCHS': 100, 'BATCH_SIZE': 4, 'GRAD_ACCUMULATION_STEPS': 2,
            'OPTIMIZER': {'lr': 1.5e-4, 'weight_decay': 0.1},
            'LR_SCHEDULER': {'warmup_epochs': 10, 'min_lr_ratio': 1e-6},
            'EARLY_STOPPING': {'patience': 20, 'metric': 'mIoU'},
            'PRETRAINED_WEIGHTS_PATH': 'pretrained_weights/swinv2_base_patch4_window12_192_22k.pth',
            'GRAD_CLIP_MAX_NORM': 0.5,
            'LOSS': {
                'alpha': 0.8,
                'beta': 0.2,
                'CLASS_WEIGHTS_DICE': [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
        },
        'MODEL': {
            'TYPE': 'DualSwinV2MonaUperNet', 'NAME': 'dual_swin_v2_mona_base',
            'DROP_PATH_RATE': 0.2, 'NUM_CLASSES': len(active_dataset['CLASSES']),
            'SWINV2': {
                'EMBED_DIM': 128, 'DEPTHS': [2, 2, 18, 2], 'NUM_HEADS': [4, 8, 16, 32],
                'WINDOW_SIZE': 16, 'PRETRAINED_WINDOW_SIZES': [12, 12, 12, 6], 'MLP_RATIO': 4.0,
                'QKV_BIAS': True, 'DROP_RATE': 0.0, 'ATTN_DROP_RATE': 0.0, 'APE': False,
                'PATCH_NORM': True, 'USE_CHECKPOINT': False,
                'LORA': {'rank': 0, 'alpha': 0},
            },
            'INTERACTION': {
                'CROSS_ATTN_IN_MONA_BLOCKS': [
                    [False, True],
                    [False, True],
                    [False] * 17 + [True],
                    [False, True]
                ],
                'CROSS_ATTN_NUM_HEADS': 8,
                'MONA_COMPLEX_GABOR_FILTERS': 24,
            },
            'ADAPTER': {
                'CONV_INPLANE': 64, 'N_POINTS': 4,
                'DEFORM_NUM_HEADS': 8,
                'INIT_VALUES': 0.0,
                'WITH_CFFN': True, 'CFFN_RATIO': 0.25, 'DEFORM_RATIO': 1.0,
                'USE_EXTRA_EXTRACTOR': True, 'GATE_INITIAL_VALUE': 0.5,
            },
            'DECODER': {
                'CHANNELS': 512, 'POOL_SCALES': (1, 2, 3, 6), 'DROPOUT_RATIO': 0.1,
            }
        }
    }
    return CONFIG