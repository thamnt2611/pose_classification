from pose_estimation.model import builder
import yaml
from easydict import EasyDict
import torch
from pose_estimation.simple_transform import SimpleTransform
from utils.transform import heatmap_to_coord_simple, perspective_transform
import json
import cv2

class PoseEstimator(object):
    """
        Using AlphaPose models: https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md#model-zoo
        In preprocessing phase, use simple transformation of AlphaPose
    """
    def __init__(self,weight_path, config_path, custom_trans_matrix=None):
        cfg = self._load_config(config_path)
        self.model = builder.build_sppe(cfg.MODEL, preset_cfg = cfg.DATA_PRESET)
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cuda:0"))) 
        self.model.eval()
        # x = torch.rand((1, 3, 256, 192))
        # traced_model = torch.jit.trace(self.model, x)
        # traced_model.save("./FastPose.jit")
        # # self.dataset_format = builder.retrieve_dataset(cfg.DATASET.TRAIN) # tra ve class dai dien cho tap dl do
        self.INPUT_SIZE = cfg.DATA_PRESET.IMAGE_SIZE
        self.NUM_JOINTS = cfg.DATA_PRESET.NUM_JOINTS
        self.transform = SimpleTransform(self.INPUT_SIZE)
        self.custom_trans_matrix = custom_trans_matrix

    def _preprocess(self, orig_img, boxes, confidences, classIDs):
        inps = torch.zeros(boxes.shape[0], 3, *self.INPUT_SIZE)
        resized_boxes = torch.zeros(boxes.shape[0], 4)
        for i, box in enumerate(boxes):
            inps[i], n_box = self.transform.test_transformation(orig_img, box)
            resized_boxes[i] = torch.FloatTensor(n_box)
        return (inps, resized_boxes)

    def predict(self, orig_img, boxes, confidences, classIDs):
        inps, resized_boxes = self._preprocess(orig_img, boxes, confidences, classIDs)
        hms = []
        for j in range(len(inps)):
            inps_j = inps[j]
            inps_j = torch.unsqueeze(inps_j, 0)
            hm_j = self.model(inps_j)
            hms.append(hm_j)
        if len(hms) == 0:
            return None
        hms = torch.cat(hms)
        hms = hms.cpu()
        preds_coords, preds_scores = self._postprocess(hms, resized_boxes)
        # result = {
        #     'pose_coords': preds_coords, # pose_num x kp_num x 2
        #     'kp_scores' : preds_scores, # pose_num x kp_num x 1
        #     'proposal_score' : torch.mean(preds_scores, 1) + confidences + 1.25 * torch.max(preds_scores, 1), # TODO: error (tensor + list)
        # }
        # n_image = cv2.warpPerspective(orig_img, self.custom_trans_matrix, (orig_img.shape[1], orig_img.shape[0]), flags=cv2.INTER_LINEAR)
        _result = []
        for k in range(hms.shape[0]):
            # print(torch.mean(preds_scores[k]), confidences[k], max(preds_scores[k]))
            _result.append(
                {
                    'keypoints':preds_coords[k],
                    'kp_score':preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + confidences[k] + 1.25 * max(preds_scores[k]),
                    'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                }
            )
        result = {
            'result': _result,
            'image': orig_img 
        }
        self._write_result_to_json(result)
        return result

    def _write_result_to_json(self, im_res):
        json_results = []
        for human in im_res['result']:
            keypoints = []
            result = {}
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            if 'box' in human.keys():
                result['box'] = human['box']
            json_results.append(result)
        with open("./result.json", "w") as f:
            f.write(json.dumps(json_results))
        # return json_results
        
        
    def _postprocess(self, hms, boxes):
        pose_coords = []
        pose_scores = []
        for i in range(hms.shape[0]):
            pred, score = heatmap_to_coord_simple(hms[i], boxes[i])
            if self.custom_trans_matrix is not None:
                for j in range(len(pred)):
                    pred[j] = perspective_transform(pred[j], self.custom_trans_matrix)
            # import pdb; pdb.set_trace()
            pose_coords.append(torch.from_numpy(pred).unsqueeze(0))
            pose_scores.append(torch.from_numpy(score).unsqueeze(0))

            # TODO: chuyen box ve goc quay tu tren xuong
            # boxes[i][0] = perspective_transform(boxes[i][0], self.custom_trans_matrix)
            # boxes[i][1] = perspective_transform(boxes[i][1], self.custom_trans_matrix)
        # import pdb; pdb.set_trace()
        preds_coords = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        return preds_coords, preds_scores
        
    def _load_config(self, config_file):
        with open(config_file) as f:
            config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
