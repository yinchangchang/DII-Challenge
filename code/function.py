# coding=utf8
#########################################################################
# File Name: function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月12日 星期三 14时28分43秒
#########################################################################

import os

from sklearn import metrics
import numpy as np

import torch

# file
import loaddata
from tools import parse
# from loaddata import data_function

args = parse.args

def save_model(p_dict, name='best.ckpt', folder='../data/models/'):
    args = p_dict['args']
    name = '{:s}-snm-{:d}-snr-{:d}-value-{:d}-trend-{:d}-cat-{:d}-lt-{:d}-size-{:d}-seed-{:d}-{:s}'.format(args.task, 
            args.split_num, args.split_nor, args.use_value, args.use_trend, 
            args.use_cat, args.last_time, args.embed_size, args.seed, name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    model = p_dict['model']
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    all_dict = {
            'epoch': p_dict['epoch'],
            'args': p_dict['args'],
            'best_metric': p_dict['best_metric'],
            'state_dict': state_dict 
            }
    torch.save(all_dict, os.path.join(folder, name))

def load_model(p_dict, model_file):
    all_dict = torch.load(model_file)
    p_dict['epoch'] = all_dict['epoch']
    # p_dict['args'] = all_dict['args']
    p_dict['best_metric'] = all_dict['best_metric']
    # for k,v in all_dict['state_dict'].items():
    #     p_dict['model_dict'][k].load_state_dict(all_dict['state_dict'][k])
    p_dict['model'].load_state_dict(all_dict['state_dict'])


def save_segmentation_results(images, segmentations, folder='../data/middle_segmentation'):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)

    # images = images.data.cpu().numpy()
    # segmentations = segmentations.data.cpu().numpy()
    images = (images * 128) + 127
    segmentations[segmentations>0] = 255
    segmentations[segmentations<0] = 0

    # print(images.shape, segmentations.shape)
    for ii, image, seg in zip(range(len(images)), images, segmentations):
        image = data_function.numpy_to_image(image)
        new_seg = np.zeros([3, seg.shape[1] * stride, seg.shape[2] * stride])
        for i in range(seg.shape[1]):
            for j in range(seg.shape[2]):
                for k in range(3):
                    new_seg[k, i*stride:(i+1)*stride, j*stride:(j+1)*stride] = seg[0,i,j]
        seg = new_seg
        seg = data_function.numpy_to_image(seg)
        image.save(os.path.join(folder, str(ii) + '_image.png'))
        seg.save(os.path.join(folder, str(ii) + '_seg.png'))


def save_middle_results(data, folder = '../data/middle_images'):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)
    numpy_data = [x.data.numpy() for x in data[1:]]
    data =  data[:1] + numpy_data
    image_names, images, word_labels, seg_labels, bbox_labels, bbox_images =  data[:6]
    images = (images * 128) + 127
    seg_labels = seg_labels*127 + 127


    for ii, name, image, seg, bbox_image in zip(range(len(image_names)), image_names, images, seg_labels, bbox_images):
        name = name.split('/')[-1]
        image = data_function.numpy_to_image(image)
        new_seg = np.zeros([3, seg.shape[1] * stride, seg.shape[2] * stride])
        # print(seg[0].max(),seg[0].min())
        for i in range(seg.shape[1]):
            for j in range(seg.shape[2]):
                for k in range(3):
                    new_seg[k, i*stride:(i+1)*stride, j*stride:(j+1)*stride] = seg[0,i,j]
        seg = new_seg
        seg = data_function.numpy_to_image(seg)
        # image.save(os.path.join(folder, name))
        # seg.save(os.path.join(folder, name.replace('image.png', 'seg.png')))
        image.save(os.path.join(folder, str(ii) + '_image.png'))
        seg.save(os.path.join(folder, str(ii) + '_seg.png'))

        for ib,bimg in enumerate(bbox_image):
            # print(bimg.max(), bimg.min(), bimg.dtype)
            bimg = data_function.numpy_to_image(bimg)
            bimg.save(os.path.join(folder, str(ii)+'_'+ str(ib) + '_bbox.png'))

def save_detection_results(names, images, detect_character_output, folder='../data/test_results/'):
    stride = args.stride

    if not os.path.exists(folder):
        os.mkdir(folder)
    # images = images.data.cpu().numpy()                                      # [bs, 3, w, h]
    images = (images * 128) + 127
    # detect_character_output = detect_character_output.data.cpu().numpy()    # [bs, w, h, n_anchors, 5+class]

    for i, name, image, bboxes in zip(range(len(names)), names, images, detect_character_output):
        name = name.split('/')[-1]

        ### 保存原图
        # data_function.numpy_to_image(image).save(os.path.join(folder, name))
        data_function.numpy_to_image(image).save(os.path.join(folder, str(i) + '_image.png'))

        detected_bbox = detect_function.nms(bboxes)
        # print([b[-1] for b in detected_bbox])
        # print(len(detected_bbox))
        image = data_function.add_bbox_to_image(image, detected_bbox)
        # image.save(os.path.join(folder, name.replace('.png', '_bbox.png')))
        image.save(os.path.join(folder, str(i) + '_bbox.png'))



def compute_detection_metric(outputs, labels, loss_outputs,metric_dict):
    loss_outputs[0] = loss_outputs[0].data
    metric_dict['metric'] = metric_dict.get('metric', []) + [loss_outputs]

def compute_segmentation_metric(outputs, labels, loss_outputs, metric_dict):
    loss_outputs[0] = loss_outputs[0].data
    metric_dict['metric'] = metric_dict.get('metric', []) + [loss_outputs]

def compute_metric(outputs, labels, time, loss_outputs,metric_dict, phase='train'):
    # loss_output_list, f1score_list, recall_list, precision_list):
    if phase != 'test':
        preds = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
    else:
        preds = np.array(outputs)

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    if time is not None:
        time = time.reshape(-1)
        assert preds.shape == time.shape
        time = time[labels>-0.5]
    assert preds.shape == labels.shape

    preds = preds[labels>-0.5]
    label = labels[labels>-0.5]

    pred = preds > 0

    assert len(pred) == len(label)

    tp = (pred + label == 2).sum()
    tn = (pred + label == 0).sum()
    fp = (pred - label == 1).sum()
    fn = (pred - label ==-1).sum()
    fp = (pred - label == 1).sum()

    metric_dict['tp'] = metric_dict.get('tp', 0.0) + tp
    metric_dict['tn'] = metric_dict.get('tn', 0.0) + tn
    metric_dict['fp'] = metric_dict.get('fp', 0.0) + fp
    metric_dict['fn'] = metric_dict.get('fn', 0.0) + fn
    loss = []
    for x in loss_outputs:
        if x == 0:
            loss.append(x)
        else:
            loss.append(x.data.cpu().numpy())
    # loss = [[x.data.cpu().numpy() for x in loss_outputs]]
    metric_dict['loss'] = metric_dict.get('loss', []) +  [loss]
    if phase != 'train':
        metric_dict['preds'] = metric_dict.get('preds', []) + list(preds)
        metric_dict['labels'] = metric_dict.get('labels', []) + list(label)
        if time is not None:
            metric_dict['times'] = metric_dict.get('times', []) + list(time)

def compute_metric_multi_classification(outputs, labels, loss_outputs, metric_dict):
    preds = outputs.data.cpu().numpy() > 0
    labels = labels.data.cpu().numpy()
    for pred, label in zip(preds, labels):
        pred = np.argmax(pred)
        tp = (pred == label ).sum()
        fn = (pred != label).sum()
        accuracy = 1.0 * tp / (tp + fn)
        metric_dict['accuracy'] = metric_dict.get('accuracy', []) + [accuracy]
    metric_dict['loss'] = metric_dict.get('loss', []) +  [[x.data.cpu().numpy() for x in loss_outputs]]


def print_metric(first_line, metric_dict, phase='train'):
    print(first_line)
    loss_array = np.array(metric_dict['loss']).mean(0)
    tp = metric_dict['tp']
    tn = metric_dict['tn']
    fp = metric_dict['fp']
    fn = metric_dict['fn']
    accuracy = 1.0 * (tp + tn) / (tp + tn + fp + fn)
    recall = 1.0 * tp / (tp + fn + 10e-20)
    precision = 1.0 * tp / (tp + fp + 10e-20)
    f1score = 2.0 * recall * precision / (recall + precision + 10e-20)


    
    loss_array = loss_array.reshape(-1)

    print('loss: {:3.4f}\t pos loss: {:3.4f}\t negloss: {:3.4f}'.format(loss_array[0], loss_array[1], loss_array[2]))
    print('accuracy: {:3.4f}\t f1score: {:3.4f}\t recall: {:3.4f}\t precision: {:3.4f}'.format(accuracy, f1score, recall, precision))
    print('\n')

    if phase != 'train':
        fpr, tpr, thr = metrics.roc_curve(metric_dict['labels'], metric_dict['preds'])
        return metrics.auc(fpr, tpr)
    else:
        return f1score

def load_all():
    fo = '../data/models'
    pre = ''
    for fi in sorted(os.listdir(fo)):
        if fi[:5] != pre:
            print
            pre = fi[:5]
        x = torch.load(os.path.join(fo, fi))
        # print x['epoch'], fi
        print x['best_metric'], fi
load_all()

