import numpy as np 
import cv2
import argparse
import abc
import time
import os
np.random.seed(42)
from ctypes import *
from .darknet_lib import DarknetLib
from collections import deque
from utils.transform import perspective_transform
class Detector(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod    
    def net(self):
        pass

    @property
    @abc.abstractmethod
    def DARKNET_INPUT_SHAPE(self):
        pass

    @property
    @abc.abstractmethod    
    def LABELS(self):
        pass

    @property
    @abc.abstractmethod    
    def INTEREST_CLASSES(self):
        pass
    
    @abc.abstractmethod
    def predict(self, image_path):
        pass

@Detector.register
class HumanDetector_OpenCV(object):
    def __init__(self, weight_path, config_path):
        self.net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.DARKNET_INPUT_SHAPE = (416, 416, 3)
        self.LABELS = open("./model/coco_labels.txt").read().strip().split("\n")
        self.INTEREST_CLASSES = [0]
        self.confidence_thres = 0.3
        self.nms_thres = 0.45

    def _get_interest_boxes(self, net_output):
        H, W = self.image_buffer.shape[:2] # (H, W)
        boxes = []
        confidences = []
        classIDs = []
        for cand in net_output:
            scores = cand[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if(confidence > 0.5) and (classID in self.INTEREST_CLASSES): 
                box = cand[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                # import pdb; pdb.set_trace()
                boxes.append(np.array([centerX, centerY, width, height]))
                classIDs.append(classID)
                confidences.append(float(confidence))
        boxes = np.cat(boxes)
        return boxes, confidences, classIDs
    
    def _preprocess_input(self):
        darknet_image = cv2.dnn.blobFromImage(self.image_buffer, 1/255.0, (self.DARKNET_INPUT_SHAPE[0], self.DARKNET_INPUT_SHAPE[1]), swapRB = True, crop = False)
        return darknet_image
    
    def predict(self, image):
        self.image_buffer = image
        darknet_image = self._preprocess_input()
        self.net.setInput(darknet_image)
        net_output = self.net.forward()
        boxes, confidences, classIDs = self._get_interest_boxes(net_output)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_thres, self.nms_thres)
        boxes = [boxes[i] for i in idxs]
        confidences = [confidences[i] for i in idxs]
        classIDs = [classIDs[i] for i in idxs]
        return boxes, confidences, classIDs

@Detector.register
class HumanDetector_Darknet(object):
    def __init__(self, weight_path, config_path, custom_transform_matrix=None):
        self.lib = DarknetLib()
        self.LABELS = open("./model/coco_labels.txt").read().strip().split("\n")
        self._load_network(config_path, weight_path)
        self.thresh = 0.5
        self.hier_thresh = 0.5
        self.nms = 0.45
        self.INTEREST_CLASSES = [0]
        self.prev_bbox_deque = deque([], maxlen=10)
        self.tracking_mem = dict()
        self.custom_transform_matrix = custom_transform_matrix # matrix for transforming in the end of postprocessing

    def _load_network(self, config_file, weights):
        """
        load model description and weights from config files
        args:
            config_file (str): path to .cfg model file
            data_file (str): path to .data model file
            weights (str): path to weights
        returns:
            network: trained model
            class_names
            class_colors
        """
        self.net = self.lib.load_net_custom(
            config_file.encode("ascii"),
            weights.encode("ascii"), 0, 1)

    def _preprocess_input(self):
        H, W = self.lib.network_height(self.net), self.lib.network_width(self.net)
        darknet_image = self.lib.make_image(W, H, 3)
        image_rgb = cv2.cvtColor(self.image_buffer, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (W, H),
                               interpolation=cv2.INTER_LINEAR)
        self.lib.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        return darknet_image

    def predict(self, image):
        # import pdb; pdb.set_trace()
        self.image_buffer = image
        darknet_image = self._preprocess_input()
        H, W = image.shape[0:2]
        n_H, n_W = self.lib.network_height(self.net), self.lib.network_width(self.net)
        scale = (float(W / n_W), float(H / n_H))
        self.lib.predict_image(self.net, darknet_image)
        pnum = pointer(c_int(0))
        detections = self.lib.get_network_boxes(self.net, darknet_image.w, darknet_image.h, 
                                            self.thresh, self.hier_thresh, None, 0, pnum, 0)
        num = pnum[0]
        self.lib.do_nms_sort(detections, num, len(self.LABELS), self.nms)

        return self._post_process(image, detections, num, scale)

    def _post_process(self, image, detections, num, scale_factor):
        scaleX, scaleY = scale_factor
        boxes, confidences, classIDs = [], [], []
        for j in range(num):
            for idx in range(len(self.LABELS)):
                if detections[j].prob[idx] > 0 and idx in self.INTEREST_CLASSES:
                    bbox = detections[j].bbox
                    bbox.x *= scaleX
                    bbox.y *= scaleY
                    bbox.w *= scaleX
                    bbox.h *= scaleY
                    if self.custom_transform_matrix is not None:
                        center_point = perspective_transform([bbox.x, bbox.y], self.custom_transform_matrix)
                        max_point = perspective_transform([bbox.x + bbox.w/2, bbox.y + bbox.h/2], self.custom_transform_matrix)
                        bbox.x, bbox.y = center_point[0], center_point[1]
                        bbox.w = (max_point[0] - center_point[0])*2
                        bbox.h = (max_point[1] - center_point[1])*2
                    boxes.append([bbox.x, bbox.y, bbox.w, bbox.h])
                    confidences.append(detections[j].prob[idx])
                    classIDs.append(idx)
        return np.array(boxes), confidences, classIDs
        # if self.custom_transform_matrix != None:
        #     image = cv2.warpPerspective(image, self.custom_trans_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        # result = {
        #     "image": image,
        #     "boxes": np.array(boxes),
        #     "confidences": confidences,
        #     "classIDs": classIDs
        # }
        # return result