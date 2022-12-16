import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils
from config import cfg

anchors = utils.get_anchors(cfg.YOLO.ANCHORS)
strides = np.array(cfg.YOLO.STRIDES)
anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
classes  = utils.read_class_names(cfg.YOLO.CLASSES)
num_class = len(classes)
iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        
def focal(target, actual, alpha=1, gamma=2):
    focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
    return focal_loss

def bbox_giou(boxes1, boxes2):


    boxes1 = tf.concat([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                        boxes1[..., :3] + boxes1[..., 3:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                        boxes2[..., :3] + boxes2[..., 3:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :3], boxes1[..., 3:]),
                        tf.maximum(boxes1[..., :3], boxes1[..., 3:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :3], boxes2[..., 3:]),
                        tf.maximum(boxes2[..., :3], boxes2[..., 3:])], axis=-1)

    boxes1_area = (boxes1[..., 3] - boxes1[..., 0]) * (boxes1[..., 4] - boxes1[..., 1]) * (boxes1[..., 5] - boxes1[..., 2])
    boxes2_area = (boxes2[..., 3] - boxes2[..., 0]) * (boxes2[..., 4] - boxes2[..., 1]) * (boxes2[..., 5] - boxes2[..., 2])

    left_up = tf.maximum(boxes1[..., :3], boxes2[..., :3])
    right_down = tf.minimum(boxes1[..., 3:], boxes2[..., 3:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
    union_area = boxes1_area + boxes2_area - inter_area
    #iou = inter_area / (union_area + 1e-5)
    iou = tf.math.divide_no_nan(inter_area, (union_area + 1e-5))
    # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

    enclose_left_up = tf.minimum(boxes1[..., :3], boxes2[..., :3])
    enclose_right_down = tf.maximum(boxes1[..., 3:], boxes2[..., 3:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1] * enclose[..., 2]
    #giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + 1e-5)
    giou = iou - 1.0 * tf.math.divide_no_nan((enclose_area - union_area), (enclose_area + 1e-5))
    # added 1e-6 in denominator to avoid generation of inf, which may cause nan loss

    return giou

def bbox_iou(boxes1, boxes2):

    boxes1_area = boxes1[..., 3] * boxes1[..., 4] * boxes1[..., 5]
    boxes2_area = boxes2[..., 3] * boxes2[..., 4] * boxes2[..., 5]

    boxes1 = tf.concat([boxes1[..., :3] - boxes1[..., 3:] * 0.5,
                        boxes1[..., :3] + boxes1[..., 3:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :3] - boxes2[..., 3:] * 0.5,
                        boxes2[..., :3] + boxes2[..., 3:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :3], boxes2[..., :3])
    right_down = tf.minimum(boxes1[..., 3:], boxes2[..., 3:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1] * inter_section[..., 2]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = 1.0 * inter_area / union_area

    return iou

def loss_layer(conv, pred, label, bboxes, anchors, stride):

    conv_shape   = conv.shape
    batch_size   = conv_shape[0]
    output_size  = conv_shape[1]
    output_depth = conv_shape[3]
    input_size   = stride * output_size
    input_depth  = stride * output_depth

    conv = layers.Reshape((output_size, output_size, output_depth,
                             anchor_per_scale, 7 + num_class))(conv)
    
    conv_raw_conf = conv[:, :, :, :, :, 6:7]
    conv_raw_prob = conv[:, :, :, :, :, 7:]

    pred_xywh     = pred[:, :, :, :, :, 0:6]
    pred_conf     = pred[:, :, :, :, :, 6:7]

    label_xywh    = label[:, :, :, :, :, 0:6]
    respond_bbox  = label[:, :, :, :, :, 6:7]
    label_prob    = label[:, :, :, :, :, 7:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, :, 3:4] * label_xywh[:, :, :, :, :, 4:5] * label_xywh[:, :, :, :, :, 5:6] / ((input_size ** 2) * input_depth) #
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]) #
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < iou_loss_thresh, tf.float32 )

    conf_focal = focal(respond_bbox, pred_conf)

    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    #print('giou_loss shape: {}'.format(giou_loss.shape))
    #print('conf_loss shape: {}'.format(conf_loss.shape))
    #print('prob_loss shape: {}'.format(prob_loss.shape))
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4,5]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4,5]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4,5]))

    return giou_loss, conf_loss, prob_loss



def custom_loss(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('giou_loss'):
    giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

    #with tf.name_scope('conf_loss'):
    conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

    #with tf.name_scope('prob_loss'):
    prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

    return giou_loss + conf_loss + prob_loss 



def loss_giou(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		):

  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('giou_loss'):
    giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

    return giou_loss 


def loss_conf(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])



    #with tf.name_scope('conf_loss'):
    conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]


    return conf_loss 




def loss_prob(label_sbbox, label_mbbox, label_lbbox, 
		true_sbbox, true_mbbox, true_lbbox,
		pred_sbbox, pred_mbbox, pred_lbbox,
		conv_sbbox, conv_mbbox, conv_lbbox
		): 
  
    #with tf.name_scope('smaller_box_loss'):
    loss_sbbox = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_sbbox,
                                anchors = anchors[0], stride = strides[0])

    #with tf.name_scope('medium_box_loss'):
    loss_mbbox = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_mbbox,
                                anchors = anchors[1], stride = strides[1])

    #with tf.name_scope('bigger_box_loss'):
    loss_lbbox = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_lbbox,
                                anchors = anchors[2], stride = strides[2])

    #with tf.name_scope('prob_loss'):
    prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

    return prob_loss 
