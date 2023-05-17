# test 2
cfg = edict({
    "batch_size": 1,
    "crop_size": 513,
    "image_mean": [103.53, 116.28, 123.675],
    "image_std": [57.375, 57.120, 58.395],
    "scales": [1.0],  # [0.5,0.75,1.0,1.25,1.75]
    'flip': True,

    'ignore_label': 255,
    'num_classes': 21,

    'model': 'deeplab_v3_s8',
    'freeze_bn': True,

    'if_png': True,
    'num_png': 3

})

# import moxing as mox
data_path = './machineLearning'
# if not os.path.exists(data_path):
#     mox.file.copy_parallel(src_url="s3://share-course/dataset/voc2012_raw/", dst_url=data_path)
cfg.data_file = data_path

# dataset
dataset = SegDataset(image_mean=cfg.image_mean,
                     image_std=cfg.image_std,
                     data_file=cfg.data_file)
dataset.get_gray_dataset()
# 路径更改
cfg.data_lst = os.path.join(cfg.data_file, 'txts/val.txt')
cfg.voc_img_dir = os.path.join(cfg.data_file, 'data/test_original')
cfg.voc_anno_gray_dir = os.path.join(cfg.data_file, 'SegmentationClassGray')

ckpt_path = './model'
# if not os.path.exists(ckpt_path):
#     mox.file.copy_parallel(src_url="s3://yyq-3/DATA/code/deeplabv3/model", dst_url=ckpt_path)     #if yours model had saved
cfg.ckpt_file = os.path.join(ckpt_path, 'deeplab_v3_s8-3_91.ckpt')
print('loading checkpoing:', cfg.ckpt_file)

net_eval(cfg)