config = {
    'base_lr': 1e-3,
    'lr_crop_size': 17,

    'TRAIN_DIR': '../../datasets/super-resolution/T91',
    'train_resampling': 256,
    'train_read_into_memory': True, # Since T91 is a small dataset, we can read them all
    'train_num_workers': 0, # Main process

    'VAL_DIR': '../../datasets/super-resolution/Set5',
    'val_resampling': 1,
    'val_read_into_memory': True,
}