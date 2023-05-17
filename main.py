# 设定参数并训练网络
from easydict import EasyDict as edict
import os
from train import train
from dataLoader import SegDataset

cfg = edict({
    "batch_size": 16,
    "crop_size": 513,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "min_scale": 0.5,
    "max_scale": 2.0,
    "ignore_label": 255,
    "num_classes": 21,
    "train_epochs": 3,

    "lr_type": 'cos',
    "base_lr": 0.0,

    "lr_decay_step": 3 * 91,
    "lr_decay_rate": 0.1,

    "loss_scale": 2048,

    "model": 'deeplab_v3_s8',
    'rank': 0,
    'group_size': 1,
    'keep_checkpoint_max': 1,
    'train_dir': 'model',

    'is_distributed': False,
    'freeze_bn': True
})

if os.path.exists(cfg.train_dir):
    shutil.rmtree(cfg.train_dir)

# 这个要改
data_path = './machineLearning'

cfg.data_file = data_path

# 输出模型路径，没事不用改
ckpt_path = './ckpt/deeplab_v3_s8-300_11.ckpt'

cfg.ckpt_file = ckpt_path

train(cfg)