#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
# ================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import utils
from config import cfg


class Dataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_type, uniform_batches=True):

        if dataset_type == 'train':
            self.annot_path = r'/home/GPU/Arthur/Code/yolo/yolov3_3d/data/dataset/liver_train.txt'
        else:
            self.annot_path = r'/home/GPU/Arthur/Code/yolo/yolov3_3d/data/dataset/liver_test.txt'

        self.dataset_type = dataset_type
        self.batch_size = cfg.YOLO.BATCH_SIZE
        self.data_aug = True if dataset_type == 'train' else False
        self.uniform_batches = uniform_batches

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.train_input_size = cfg.YOLO.INPUT_SIZE
        self.train_input_depth = cfg.YOLO.INPUT_DEPTH
        self.strides = cfg.YOLO.STRIDES
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.max_bbox_per_scale = cfg.YOLO.MAX_BBOX_PER_SCALE

        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)

        if self.uniform_batches:
            self.num_batches = self.num_samples // self.batch_size
        else:
            self.num_batches = math.ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        #self.train_input_size = random.choice(self.train_input_sizes)
        #self.train_output_sizes = self.train_input_size // self.strides
        self.train_output_sizes = [
            self.train_input_size // x for x in self.strides]
        self.train_output_depths = [
            self.train_input_depth // x for x in self.strides]

        batch_image = np.zeros(
            (self.batch_size, self.train_input_size, self.train_input_size, self.train_input_depth, 1))

        batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                      self.train_output_depths[0],
                                      self.anchor_per_scale, 7 + self.num_classes))
        batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                      self.train_output_depths[1],
                                      self.anchor_per_scale, 7 + self.num_classes))
        batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                      self.train_output_depths[2],
                                      self.anchor_per_scale, 7 + self.num_classes))

        batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 6))
        batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 6))
        batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 6))

        num = 0
        # if self.batch_count < self.num_batchs:
        self.batch_count = idx
        while num < self.batch_size:
            index = self.batch_count * self.batch_size + num
            if index >= self.num_samples:
                index -= self.num_samples
            annotation = self.annotations[index]
            image, bboxes = self.parse_annotation(annotation)
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                bboxes)

            batch_image[num, :, :, :] = image
            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes
            num += 1

        return [batch_image,
                batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
                batch_sbboxes, batch_mbboxes, batch_lbboxes]

        '''
        return batch_image,
        [batch_label_sbbox, batch_label_mbbox, batch_label_lbbox,
         batch_sbboxes, batch_mbboxes, batch_lbboxes]
        '''

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip()
                           for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations

    def horizontal_flip(self, image, bboxes):

        _, w, _, _ = image.shape
        image = image[:, ::-1, :, :]
        bboxes[:, [0, 3]] = w - bboxes[:, [3, 0]]

        return image, bboxes

    def vertical_flip(self, image, bboxes):

        h, _, _, _ = image.shape
        image = image[::-1, :, :, :]
        bboxes[:, [1, 4]] = h - bboxes[:, [4, 1]]

        return image, bboxes

    def rotk1(self, image, bboxes):

        d, _, _, _ = image.shape
        #image = image[::-1, :, :]
        image = np.rot90(image, k=1)
        bboxes[:, 0] = d - bboxes[:, 4] #2->3, 3->4
        bboxes[:, 1] = bboxes[:, 0]
        bboxes[:, 3] = d - bboxes[:, 1]
        bboxes[:, 4] = bboxes[:, 3]

        return image, bboxes

    def rotk3(self, image, bboxes):

        d, _, _, _ = image.shape
        #image = image[::-1, :, :]
        image = np.rot90(image, k=3)
        bboxes[:, 0] = bboxes[:, 1]
        bboxes[:, 1] = d - bboxes[:, 3]
        bboxes[:, 3] = bboxes[:, 4]
        bboxes[:, 4] = d - bboxes[:, 0]

        return image, bboxes

    def random_augment(self, image, bboxes):
        flip_num = random.randint(1, 8)
        if flip_num == 1:
            image, bboxes = self.horizontal_flip(image, bboxes)
        elif flip_num == 2:
            image, bboxes = self.vertical_flip(image, bboxes)
        elif flip_num == 3:
            image, bboxes = self.rotk1(image, bboxes)
        elif flip_num == 4:
            image, bboxes = self.rotk3(image, bboxes)
        elif flip_num == 5:
            image, bboxes = self.horizontal_flip(image, bboxes)
            image, bboxes = self.rotk1(image, bboxes)
        elif flip_num == 6:
            image, bboxes = self.horizontal_flip(image, bboxes)
            image, bboxes = self.rotk3(image, bboxes)
        elif flip_num == 7:
            image, bboxes = self.horizontal_flip(image, bboxes)
            image, bboxes = self.vertical_flip(image, bboxes)
        return image, bboxes

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 3]] = w - bboxes[:, [3, 0]]

        return image, bboxes

    def random_vertical_flip(self, image, bboxes):

        if random.random() < 0.5:
            h, _, _, _ = image.shape
            image = image[::-1, :, :]
            bboxes[:, [1, 4]] = h - bboxes[:, [4, 1]]

        return image, bboxes

    def random_rotk1(self, image, bboxes):

        if random.random() < 0.5:
            d, _, _, _ = image.shape
            #image = image[::-1, :, :]
            image = np.rot90(image, k=1)
            bboxes[:, 0] = d - bboxes[:, 4]
            bboxes[:, 1] = bboxes[:, 0]
            bboxes[:, 3] = d - bboxes[:, 1]
            bboxes[:, 4] = bboxes[:, 3]

        return image, bboxes

    def random_rotk3(self, image, bboxes):

        if random.random() < 0.5:
            d, _, _, _ = image.shape
            #image = image[::-1, :, :]
            image = np.rot90(image, k=3)
            bboxes[:, 0] = bboxes[:, 1]
            bboxes[:, 1] = d - bboxes[:, 3]
            bboxes[:, 3] = bboxes[:, 4]
            bboxes[:, 4] = d - bboxes[:, 0]

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        #image = np.array(cv2.imread(image_path))

        image = np.load(image_path)
        #image = np.sum(image, axis=-1, keepdims=True)
        if len(image.shape) == 3:
            image = image[..., np.newaxis]
        image = image * 255
        #image = np.repeat(image, 3, axis=-1)

        bboxes = np.array(
            [list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        if self.data_aug:
            #image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            #image, bboxes = self.random_vertical_flip(np.copy(image), np.copy(bboxes))
            
            image, bboxes = self.random_horizontal_flip(image, bboxes)
            image, bboxes = self.random_vertical_flip(image, bboxes)
            
            image, bboxes = self.random_rotk1(image, bboxes)
            image, bboxes = self.random_rotk3(image, bboxes)
            #image, bboxes = self.random_augment(image, bboxes)
            #image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            #image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        #Padding for the image. Not needed in this case.
        '''
        image, bboxes = utils.image_preprocess(
            np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        '''
        
        updated_bb = []
        for bb in bboxes:
            x1, y1, z1, x2, y2, z2, cls_label = bb

            if x2 <= x1 or y2 <= y1 or z2 <= z1:
                # dont use such boxes as this may cause nan loss.
                continue

            x1 = int(np.clip(x1, 0, image.shape[1]))
            y1 = int(np.clip(y1, 0, image.shape[0]))
            z1 = int(np.clip(y1, 0, image.shape[2]))
            x2 = int(np.clip(x2, 0, image.shape[1]))
            y2 = int(np.clip(y2, 0, image.shape[0]))
            z2 = int(np.clip(y2, 0, image.shape[2]))
            # clipping coordinates between 0 to image dimensions as negative values
            # or values greater than image dimensions may cause nan loss.
            updated_bb.append([x1, y1, z1, x2, y2, z2, cls_label])

        return image, np.array(updated_bb)

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 3] * boxes1[..., 4] #x, y, z, w, h, d
        boxes2_area = boxes2[..., 3] * boxes2[..., 4] #x, y, z, w, h, d

        boxes1 = np.concatenate([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                                boxes1[..., :3] + boxes1[..., 3:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                                boxes2[..., :3] + boxes2[..., 3:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :3], boxes2[..., :3])
        right_down = np.minimum(boxes1[..., 3:], boxes2[..., 3:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / (union_area + 1e-6)
        # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], 
			   self.train_output_depths[i],
			   self.anchor_per_scale,
                           7 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 6))
                       for _ in range(3)]
        bbox_count = np.zeros((3,)) #??????

        for bbox in bboxes:
            bbox_coor = bbox[:6]
            bbox_class_ind = bbox[6]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0

            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [(bbox_coor[3:] + bbox_coor[:3]) * 0.5, bbox_coor[3:] - bbox_coor[:3]], axis=-1)


            bbox_xywh_scaled = 1.0 * \
                bbox_xywh[np.newaxis, :] / \
                np.array(self.strides)[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 6))
                anchors_xywh[:, 0:3] = np.floor(
                    bbox_xywh_scaled[i, 0:3]).astype(np.int32) + 0.5
                anchors_xywh[:, 3:6] = self.anchors[i]

                iou_scale = self.bbox_iou(
                    bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind, zind = np.floor(
                        bbox_xywh_scaled[i, 0:3]).astype(np.int32)
                    xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)
                    yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)
                    zind = np.clip(zind, 0, self.train_output_depths[i] - 1)
                    # This will mitigate errors generated when the location computed by this is more the grid cell location.
                    # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both.
                    # But sometimes the coomputation makes it 52 and then it will try to find that location in label array
                    # which is not present and throws error during training.

                    label[i][yind, xind, zind, iou_mask, :] = 0
                    label[i][yind, xind, zind, iou_mask, 0:6] = bbox_xywh
                    label[i][yind, xind, zind, iou_mask, 6:7] = 1.0
                    label[i][yind, xind, zind, iou_mask, 7:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :6] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind, zind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:3]).astype(np.int32)
                xind = np.clip(xind, 0, self.train_output_sizes[i] - 1)
                yind = np.clip(yind, 0, self.train_output_sizes[i] - 1)
                zind = np.clip(zind, 0, self.train_output_depths[i] - 1)
                # This will mitigate errors generated when the location computed by this is more the grid cell location.
                # e.g. For 52x52 grid cells possible values of xind and yind are in range [0-51] including both.
                # But sometimes the coomputation makes it 52 and then it will try to find that location in label array
                # which is not present and throws error during training.

                label[best_detect][yind, xind, zind, best_anchor, :] = 0
                label[best_detect][yind, xind, zind, best_anchor, 0:6] = bbox_xywh
                label[best_detect][yind, xind, zind, best_anchor, 6:7] = 1.0
                label[best_detect][yind, xind, zind, best_anchor, 7:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] %
                               self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :6] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
