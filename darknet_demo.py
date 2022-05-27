import numpy as np 
import cv2
import argparse
import abc
import time
import os
import torch
np.random.seed(42)
from ctypes import *
from detection.detector import HumanDetector_Darknet, HumanDetector_OpenCV
from pose_estimation.pose_estimator import PoseEstimator
import yaml
from easydict import EasyDict
from utils.visualize import vis_frame_simple
from utils.input_queue import ImageQueue
from utils.transform import get_custome_transform_matrix, perspective_transform
from data_format import FrameData, ObjectInfo
from tracking.tracker import Tracker
from tracking.kalman_tracker import KalmanFilterTracker
from classification.pose_classifier import PoseClassifier, CocoPoseEncoder

class PoseApp:
    def __init__(self, args):
        # self.custome_transform_matrix = get_custome_transform_matrix()
        self.custome_transform_matrix = None
        self.detector = HumanDetector_Darknet(args.det_weight, args.det_cfg, args.det_labels, self.custome_transform_matrix)
        self.pose_estimator = PoseEstimator(args.pe_weight, args.pe_cfg)
        pose_encoder = CocoPoseEncoder()
        self.pose_classifier = PoseClassifier(args.pose_clf_weight, args.pose_names, pose_encoder, 17)
        self.tracker = KalmanFilterTracker()
        self.input_queue = ImageQueue(args.input_path)
        self.data_pipeline = [] # list DataItem

    def run(self):
        self.input_queue.start_read_input()
        import time
        frame_idx = 0
        while(True):
            app_item = FrameData()
            image = self.input_queue.get()
            if self.custome_transform_matrix is None:
                app_item.image = image
            else:
                app_item.image = cv2.warpPerspective(image, self.custome_transform_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
            boxes, confidences, classIDs = self.detector.predict(image)
            pose_results = self.pose_estimator.predict(image, boxes, confidences, classIDs)
            for i in range(len(confidences)):
                obj_info = ObjectInfo()
                obj_info.box = boxes[i]
                obj_info.confidence = confidences[i]
                obj_info.class_id = classIDs[i]
                obj_info.frame_id = frame_idx
                obj_info.pose_name = self.pose_classifier.predict(pose_results['result'][i])
                app_item.object_infos.append(obj_info)
            self.tracker.track_object(app_item.object_infos)
            for object_info in app_item.object_infos:
                # print("Frame: ", frame_idx, " - Object: ", object_info.track_id, " - Velocity: {:2f}".format(object_info.velocity), " - Pose: ", object_info.pose_name)
                # print("Frame: ", frame_idx)
                # print(object_info.box)
                print("Frame: ", frame_idx, " - Object: ", object_info.track_id, " - Corrected box: ", object_info.corrected_box, " - Velocity: ", object_info.velocity)
            frame_idx += 1
            n_image = self.draw_result(app_item, pose_results)
            cv2.imshow("Pose result", n_image)
            if (self.input_queue.ips != 0):
                cv2.waitKey(25)
            else:
                cv2.waitKey(0)

    def draw_result(self, app_item, pose_result, time = 0):
        image = app_item.image
        from utils.visualize import vis_frame_simple
        image = vis_frame_simple(pose_result, self.pose_estimator.NUM_JOINTS)
        object_infos = app_item.object_infos
        for object_info in object_infos:
            box = object_info.box
            corrected_box = object_info.corrected_box
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            color = [0, 0, 0] # more contrast than the original color
            xmin = int(round(x - (w / 2)))
            xmax = int(round(x + (w / 2)))
            ymin = int(round(y - (h / 2)))
            ymax = int(round(y + (h / 2)))

            x, y, w, h = int(corrected_box[0]), int(corrected_box[1]), int(corrected_box[2]), int(corrected_box[3])
            c_color = [0, 255, 0] # more contrast than the original color
            c_xmin = int(round(x - (w / 2)))
            c_xmax = int(round(x + (w / 2)))
            c_ymin = int(round(y - (h / 2)))
            c_ymax = int(round(y + (h / 2)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.rectangle(image, (c_xmin, c_ymin), (c_xmax, c_ymax), c_color, 2)
            text = "obj {} v: {:.2f} {}".format(object_info.track_id, object_info.velocity, object_info.pose_name)
            # text = "{}: {:.4f}".format(self.detector.LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        
        cv2.putText(image, "Time: " + str(time), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_path", "--input_path", default = "./inputs/test_video_1.mp4")
    parser.add_argument("-det_weight", "--det_weight", default = "./detection/model/yolov3-spp.weights") # det: detection
    parser.add_argument("-det_cfg", "--det_cfg", default = "./detection/model/yolov3-spp.cfg") 
    parser.add_argument("-det_labels", "--det_labels", default = "./detection/model/coco_labels.txt") 
    parser.add_argument("-pe_weight", "--pe_weight", default = "./pose_estimation/pretrained_models/fast_421_res50-shuffle_256x192.pth")
    parser.add_argument("-pe_cfg", "--pe_cfg", default = "./pose_estimation/configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml") #pe: pose_estimation
    parser.add_argument("-pose_clf_weight", "--pose_clf_weight", default = "./classification/weight/nn_model.pt")
    parser.add_argument("-pose_names", "--pose_names", default = "./classification/weight/nn_model_classes.txt")
    
    args = parser.parse_args()
    app = PoseApp(args)
    app.run()
