# coding=utf8
#########################################################################
# File Name: data_function.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: 2019年06月12日 星期三 11时28分13秒
#########################################################################

import os
import sys
import time
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter

from tools import parse
args = parse.args

def add_text_to_img(img, text, size, font, color, place):
    imgdraw = ImageDraw.Draw(img)
    imgfont = ImageFont.truetype(font,size=size)
    imgdraw.text(place, text, fill=color, font=imgfont)
    return img

def image_to_numpy(image):
    image = np.array(image)
    image = image.transpose(2, 0, 1)
    return image

def numpy_to_image(image):
    image = image.transpose(1, 2, 0).astype(np.uint8)
    return Image.fromarray(image)

def add_line(bbox_image, bbox, gray=128, proposal=0):

    # print(bbox, bbox_image.shape) 

    sx,sy,ex,ey = bbox[:4]
    _,x,y = bbox_image.shape # 3, 64, 512

    if not proposal:
        assert sx <= x
        assert ex <= x
        assert sy <= y
        assert ey <= y

    n = 2
    bbox_image[:, sx:ex, sy-n:sy+n] = gray
    bbox_image[:, sx:ex, ey-n:ey+n] = gray
    bbox_image[:, sx-n:sx+n, sy:ey] = gray
    bbox_image[:, ex-n:ex+n, sy:ey] = gray
    return bbox_image

def add_bbox_to_image(image, detected_bbox):
    words = args.words

    image = np.zeros_like(image) + 255
    image = numpy_to_image(image)
    for bbox in detected_bbox:
        bbox = [int(x) for x in bbox[1:]]
        # size = int((bbox[2] + bbox[3] - bbox[0] - bbox[0]) / 2)
        size = 16
        place = (int(bbox[1]/2 + bbox[3]/2), int(bbox[0]/2+bbox[2]/2))
        image = add_text_to_img(image, words[bbox[-1]], size, '../files/ttf/simsun.ttf', (0,0,0), place)
    return image

def test_label(image_file, seg_file, bbox_file, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    image = Image.open(image_file).convert('RGB')
    seg = Image.open(seg_file)
    image.save(os.path.join(save_folder, '_image.png'))
    seg.save(os.path.join(save_folder, '_seg.png'))

    bbox_image = image_to_numpy(image)
    bbox_label = json.load(open(bbox_file))
    for bbox in bbox_label:
        bbox_image = add_line(bbox_image, bbox)
    image = numpy_to_image(bbox_image)
    image.save(os.path.join(save_folder, '_bbox.png'))

def generate_bbox_seg(image, font_place, font_size, font_list):
    '''
    只生成框位置坐标
    '''
    imgh,imgw = image.size
    font_num = len(font_list)

    # 生成分割label
    seg_label = np.zeros((3, image.size[1], image.size[0]), dtype=np.uint8) + 255
    sy = font_place[0]
    ey = sy + font_size * font_num
    sx = font_place[1]
    ex = sx + font_size
    seg_label[:, sx:ex, sy:ey] = 128
    # seg_label = seg_label.transpose((1,0,2))
    # seg_label = Image.fromarray(seg_label)
    seg_label = numpy_to_image(seg_label)

    # 生成bbox label
    bbox_label = []
    for i, font in enumerate(font_list):
        sx = font_place[0] + font_size * i
        ex = sx + font_size
        sy = font_place[1]
        ey = sy + font_size
        bbox_label.append([sy,sx,ey,ex,font])

    # 生成bbox_image
    # bbox_image = np.zeros((3, image.size[0], image.size[1]), dtype=np.uint8) + 255
    bbox_image = image_to_numpy(image)
    for bbox in bbox_label:
        bbox_image = add_line(bbox_image, bbox)
    bbox_image = numpy_to_image(bbox_image)


    return bbox_label, seg_label, bbox_image


def generate_bbox_label(image, font_place, font_size, font_num, args, image_size):
    '''
    根据anchors生成监督信息
    '''
    imgh,imgw = image.size
    seg_label = np.zeros((int(image_size[0]/2), int(image_size[1]/2)), dtype=np.float32)
    sx = float(font_place[0]) / image.size[0] * image_size[0]
    ex = sx + float(font_size) / image.size[0] * image_size[0] * font_num
    sy = float(font_place[1]) / image.size[1] * image_size[1]
    ey = sy + float(font_size) / image.size[1] * image_size[1]
    seg_label[int(sx/2):int(ex/2), int(sy/2):int(ey/2)] = 1
    seg_label = seg_label.transpose((1,0))

    bbox_label = np.zeros((
        int(image_size[0]/args.stride),  # 16
        int(image_size[1]/args.stride),  # 16
        len(args.anchors),          # 4
        4                           # dx,dy,dd,c
        ), dtype=np.float32)
    fonts= []
    for i in range(font_num):
        x = font_place[0] + font_size/2. + i * font_size
        y = font_place[1] + font_size/2.
        h = font_size
        w = font_size

        x = float(x) * image_size[0] / imgh
        h = float(h) * image_size[0] / imgh
        y = float(y) * image_size[1] / imgw
        w = float(w) * image_size[1] / imgw
        fonts.append([x,y,h,w])

    # print bbox_label.shape
    for ix in range(bbox_label.shape[0]):
        for iy in range(bbox_label.shape[1]):
            for ia in range(bbox_label.shape[2]):
                proposal = [ix*args.stride + args.stride/2, iy*args.stride + args.stride/2, args.anchors[ia]]
                iou_fi = []
                for fi, font in enumerate(fonts):
                    iou = comput_iou(font, proposal)
                    iou_fi.append((iou, fi))
                max_iou, max_fi = sorted(iou_fi)[-1]
                if max_iou > 0.5:
                    # 正例
                    dx = (font[0] - proposal[0]) / float(proposal[2])
                    dy = (font[1] - proposal[1]) / float(proposal[2])
                    fd = max(font[2:])
                    dd = np.log(fd / float(proposal[2]))
                    # bbox_label[ix,iy,ia] = [dx, dy, dd, 1]
                    bbox_label[ix,iy,ia] = [dx, dy, dd, 1]
                elif max_iou > 0.25:
                    # 忽略
                    bbox_label[ix,iy,ia,3] = 0
                else:
                    # 负例
                    bbox_label[ix,iy,ia,3] = -1
    # 这里有一个transpose操作
    bbox_label = bbox_label.transpose((1,0,2,3))


                # 计算anchor信息
    return bbox_label, seg_label

def augment(image, seg, bbox, label):
    return image, seg, bbox, label

def random_select_indices(indices, n=10):
    indices = np.array(indices)
    # print('initial shape', indices.shape) 
    indices = indices.transpose(1,0)
    # print('change shape', indices.shape) 
    np.random.shuffle(indices)
    indices = indices[:n]
    # print('select ', indices.shape) 
    indices = indices.transpose(1,0)
    # print('change shape', indices.shape) 
    # indices = tuple(indices)
    return tuple(indices)



# test_label( '../../data/generated_images/1.png', '../../data/generated_images/1_seg.png', '../../data/generated_images/1_bbox.json', '../../data/test/')
