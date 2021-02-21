'''
This script contains the list of face detectors. FaceDetector class is the parent class to which you can
customise how you want to carry out the face detection with
'''

import cv2
import numpy as np
import os
import src.facenet as facenet
import tensorflow as tf
import src.align.detect_face as detect_face

class FaceDetector:
    '''
    The basic class for detections
    '''

    def __init__(self, min_size=20, crop_margin=44, crop_shape=(160, 160)):
        self.MIN_SIZE = min_size
        self.CROP_MARGIN = crop_margin
        self.CROP_SHAPE = crop_shape

    def extract_crops(self, im_array, bbs, only_first=False):
        crops = []
        img_shape = im_array.shape
        # returns empty list if there are no detections
        if len(bbs) > 0:
            for det in bbs:
                print("Det", det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-self.CROP_MARGIN/2, 0)
                bb[1] = np.maximum(det[1]-self.CROP_MARGIN/2, 0)
                bb[2] = np.minimum(det[2]+self.CROP_MARGIN/2, img_shape[1])
                bb[3] = np.minimum(det[3]+self.CROP_MARGIN/2, img_shape[0])
                cropped = im_array[bb[1]:bb[3],bb[0]:bb[2],:]
                aligned = cv2.resize(cropped, self.CROP_SHAPE, interpolation=cv2.INTER_LINEAR)
                prewhitened = facenet.prewhiten(aligned) # preprocessing on the image that is supposed to increase accuracy

                cv2.imshow('IMAGE', prewhitened)
                cv2.waitKey(1)
                print('Shape', prewhitened.shape)
                print('dtype', prewhitened.dtype)
                crops.append(prewhitened)

                # instantly returns if only one box is desired
                if only_first:
                    return prewhitened
            crops = np.stack(crops)

        return crops
    
    def draw_detections(self, im_array, bbs, colour=(255, 0, 0)):
        annotated = im_array.copy()
        if isinstance(bbs, list):
            return im_array
        for bb in bbs:
            xmin, ymin, xmax, ymax, score = bb
            ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
            annotated = cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color=colour, thickness=1)
            annotated = cv2.putText(annotated, 'Confidence: %.2d' % (score), (xmin, ymin+20), \
                cv2.FONT_HERSHEY_PLAIN, 1, color=(255,255,255))
        return annotated

class MTCNN(FaceDetector):

    def __init__(self, min_size=20, crop_margin=44, crop_shape=(160, 160), gpu_mem_ratio=0.6, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
        # assigning variables
        super().__init__(min_size, crop_margin, crop_shape)
        self.THRESHOLDS = thresholds
        self.FACTOR = factor

        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_ratio)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, None)

    def get_faces(self, im_array):
        bounding_boxes, _ =detect_face.detect_face(im_array, self.MIN_SIZE, self.pnet, self.rnet, \
            self.onet, self.THRESHOLDS, self.FACTOR)
        return bounding_boxes


class Cascader(FaceDetector):
    '''
    Yo En Hui you can follow my example above to adapt it such that you can also
    detect using the cascader. the concept should be the same
    '''
    def __init__(self, min_size=20, crop_margin=44, crop_shape=(160, 160)):
        super().__init__(min_size, crop_margin, crop_shape)