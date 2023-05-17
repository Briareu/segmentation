import os
import sys

# 需要更改成我的代码文件路径
sys.path.insert(0, './machineLearning')  # your code path

from easydict import EasyDict as edict
from PIL import Image
import PIL

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
import numpy as np
import cv2
# import moxing as mox

from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)


def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w


def eval_batch(args, eval_net, img_lst, crop_size=513, flip=True):
    result_lst = []
    batch_size = len(img_lst)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    resize_hw = []
    for l in range(batch_size):
        img_ = img_lst[l]
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        batch_img[l] = img_
        resize_hw.append([resize_h, resize_w])

    batch_img = np.ascontiguousarray(batch_img)
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        result_lst.append(probs_)

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=513, flip=True):
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    # print(sizes_)
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk

# 颜色与类别的对应字典，也许我们只需要backround和病灶（edge）？
# num_class = {0: 'background', 1 : 'mask'}
# num_color = {0: 'black', 1: 'white'}

# The color source: print(list(colors.cnames.keys()))
# print(list(colors.cnames.keys()))
num_class = {0: 'background', 1: 'mask'}

num_color = {0: 'black', 1: 'white'}

color_dic = [num_color[k] for k in sorted(num_color.keys())]
bounds = list(range(2))
cmap = mpl.colors.ListedColormap(color_dic)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


def num_to_ClassAndColor(num_list):
    color_ = []
    class_ = []
    for num in num_list:
        color_.append(num_class[num])
        class_.append(num_color[num])
    return color_, class_


def net_eval(args):
    # network
    if args.model == 'deeplab_v3_s16':
        network = DeepLabV3('eval', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'deeplab_v3_s8':
        network = DeepLabV3('eval', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    eval_net = BuildEvalNetwork(network)

    # load model
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)

    # data list
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    for i, line in enumerate(img_lst):
        id_ = line.strip()
        # 这个要改吧
        img_path = os.path.join(cfg.voc_img_dir, id_ + '.png')
        msk_path = os.path.join(cfg.voc_anno_gray_dir, id_ + '.png')

        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        if args.if_png:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)

            height, weight = batch_res[0].shape
            batch_msk_lst[0][batch_msk_lst[0] == args.ignore_label] = 0

            plt.figure(figsize=(3 * weight / 1024 * 10, 2 * height / 1024 * 10))
            plt.subplot(1, 3, 1)
            image = Image.open(img_path)
            plt.imshow(image)

            plt.subplot(1, 3, 2)
            plt.imshow(image)
            plt.imshow(batch_res[0], alpha=0.8, interpolation='none', cmap=cmap, norm=norm)

            plt.subplot(1, 3, 3)
            plt.imshow(image)
            plt.imshow(batch_msk_lst[0], alpha=0.8, interpolation='none', cmap=cmap, norm=norm)
            plt.show()

            prediction_num = np.unique(batch_res[0])
            real_num = np.unique(batch_msk_lst[0])

            prediction_color, prediction_class = num_to_ClassAndColor(prediction_num)
            print('prediction num:', prediction_num)
            print('prediction color:', prediction_color)
            print('prediction class:', prediction_class)
            real_color, real_class = num_to_ClassAndColor(real_num)
            print('groundtruth num:', real_num)
            print('groundtruth color:', real_color)
            print('groundtruth class:', real_class)
            batch_img_lst = []
            batch_msk_lst = []
            if i < args.num_png - 1:
                continue
            else:
                return

        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)
            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            if (i + 1) % 100 == 0:
                print('processed {} images'.format(i + 1))
        image_num = i

    if bi > 0:
        batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size, flip=args.flip)
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        if (i + 1) % 100 == 0:
            print('processed {} images'.format(image_num + 1))

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IoU', np.nanmean(iu))