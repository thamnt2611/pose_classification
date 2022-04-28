import numpy as np 
import cv2
import argparse
import abc
import time
import os
np.random.seed(42)
from ctypes import *
from detection.detector import HumanDetector_Darknet, HumanDetector_OpenCV
from pose_estimation.pose_estimator import PoseEstimator
import yaml
from easydict import EasyDict
from utils.visualize import vis_frame_simple
from utils.input_queue import ImageQueue
from utils.transform import get_custome_transform_matrix, perspective_transform
from data_format import FrameData, ObjectData
from tracking.tracker import Tracker
class PoseApp:
    def __init__(self, input_path):
        # self.custome_transform_matrix = get_custome_transform_matrix()
        self.custome_transform_matrix = None
        self.detector = HumanDetector_Darknet("./model/yolov3-spp.weights", "./model/yolov3-spp.cfg", self.custome_transform_matrix)
        self.pose_estimator = PoseEstimator("./pose_estimation/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", "./pose_estimation/pretrained_models/halpe26_fast_res50_256x192.pth")
        self.tracker = Tracker()
        self.input_queue = ImageQueue(input_path)
        self.BOX_COLORS = np.random.randint(0, 255, size=(len(self.detector.LABELS), 3), dtype="uint8")
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
            for i in range(len(confidences)):
                obj_info = ObjectData()
                obj_info.box = boxes[i]
                obj_info.confidence = confidences[i]
                obj_info.class_id = classIDs[i]
                obj_info.frame_id = frame_idx
                app_item.object_infos.append(obj_info)
            tracking_ids = self.tracker.object_tracking(app_item.object_infos)
            # for i in range(len(confidences)):
            #     app_item.object_infos[i].track_id = tracking_ids[i]
            for object_info in app_item.object_infos:
                if(object_info.track_id == 10):
                    print("Frame: ", frame_idx, " - Object: ", object_info.track_id, " - Velocity: ", object_info.velocity)
            frame_idx += 1
            n_image = self._show_detection_result(app_item)
            cv2.imshow("Pose result", n_image)
            if (self.input_queue.ips != 0):
                cv2.waitKey(25)
            else:
                cv2.waitKey(0)
            # pose_result = self.pose_estimator.predict(image, detections)
            # if (pose_result is None):
            #     continue
            # img = self._show_detection_result(image, boxes, confidences, classIDs)
            
            
        
    def _show_pose_estimation_result(self, pose_result):
        from utils.visualize import vis_frame_simple
        img = vis_frame_simple(pose_result, self.pose_estimator.NUM_JOINTS)
        return img

    def _show_detection_result(self, app_item, time = 0):
        image = app_item.image
        object_infos = app_item.object_infos
        for object_info in object_infos:
            box = object_info.box
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            # color = [int(c) for c in self.BOX_COLORS[object_info.class_id]]
            color = [0, 0, 0] # more contrast than the original color
            xmin = int(round(x - (w / 2)))
            xmax = int(round(x + (w / 2)))
            ymin = int(round(y - (h / 2)))
            ymax = int(round(y + (h / 2)))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            text = "obj {} v: {:.2f}".format(object_info.track_id, object_info.velocity)
            # text = "{}: {:.4f}".format(self.detector.LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.putText(image, "Time: " + str(time), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return image

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-input", "--input", required = True)
    # parser.add_argument("-detection_model", "--det_model", required = True)
    # parser.add_argument("-detection_config", "--det_cfg", required = True)
    # parser.add_argument("-pe_config", "--pe_cfg", required = True)
    # args = parser.parse_args()

    # human_detector = HumanDetector_OpenCV(args.weight, args.cfg)

    
    # app = DetectionApp(human_detector, args.input)
    # app.run()

    # pe_cfg = app.load_config(args.pe_cfg)
    input_path = "./inputs/test_video_1.mp4"
    app = PoseApp(input_path)
    app.run()
