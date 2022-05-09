import glob
from pose_estimation.pose_estimator import PoseEstimator
import cv2
import numpy as np
from utils.visualize import vis_frame_simple
import torch
import json
# class_names = ["chair", "cobra", "dog", "tree", "warrior"]
class_names = ["run", "stand", "walk"]

class DataLoader(object):
    def __init__(self):
        self.data = dict()

    def load_dataset(self):
        for class_name in class_names:
            file_paths = []
            for file in glob.glob("classification/data/{}/*".format(class_name)):
                file_paths.append(file)
            self.data[class_name] = file_paths

if __name__=="__main__":
    data_loader = DataLoader()
    data_loader.load_dataset()
    estimator = PoseEstimator("./pose_estimation/configs/coco/resnet/256x192_res50_lr1e-3_1x-duc.yaml", "./pose_estimation/pretrained_models/fast_421_res50-shuffle_256x192.pth")
    with open("pose_data_17.json", "w+") as f:
        for class_name in class_names:
            for file_name in data_loader.data[class_name]:
                image = cv2.imread(file_name)
                w, h = image.shape[1], image.shape[0]
                boxes = np.array([[w / 2, h / 2, w, h]])
                confidences = [1.0]
                classIDs = [0]
                result = estimator.predict(image, boxes, confidences, classIDs)
                assert len(result['result']) == 1
                pose_result = result['result'][0]
                pose = {}
                pose['image_path'] = file_name
                pose['keypoints'] = pose_result['keypoints'].tolist()
                print(len(pose['keypoints']))
                pose['kp_score'] = pose_result['kp_score'].tolist()
                pose['proposal_score'] = pose_result['proposal_score'].tolist()
                pose['box'] = pose_result['box']
                # new_result['result'].append(pose)
                pose['classname'] = class_name
                f.write(json.dumps(pose) + "\n")
