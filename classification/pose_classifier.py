import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
class NN_Model(nn.Module):
  def __init__ (self, num_keypoints, num_classes):
    super(NN_Model, self).__init__()
    self.linear_1 = nn.Linear(num_keypoints*2, 256)
    self.dropout_1 = nn.Dropout(0.2)
    self.linear_2 = nn.Linear(256, 128)
    self.dropout_2 = nn.Dropout(0.2)
    self.linear_3 = nn.Linear(128, num_classes)
    self.activate_fcn = nn.Tanh()

  def forward(self, x):
    # x : normalized keypoints
    # print(x.shape)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.linear_1(x)
    x = F.relu(x)
    x = self.linear_2(x)
    x = F.relu(x)
    x = self.linear_3(x)
    return x

class CocoPoseEncoder(object):
    """
        Convert pose in coco format into vector
        Apply centering and normalization
    """
    def __init__(self):
        self.NUM_JOINTS = 17

    def calculate_distance(self, a, b):
        dist = torch.linalg.vector_norm(a - b, dim = -1)
        return dist
    
    def get_pose_size(self, keypoints):
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        center_shoulder = 0.5 * left_shoulder + 0.5 * right_shoulder
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        center_body = 0.5 * left_hip + 0.5 * right_hip
        torso_size = self.calculate_distance(center_shoulder, center_body) * 2.5
        center_bodies = center_body.expand(keypoints.shape[0], 2)
        dist = self.calculate_distance(center_bodies, keypoints)
        pose_size = torch.max(torch.max(dist), torso_size)
        return pose_size

    def encode(self, keypoints):
        # normalize keypoints 
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        center_body = 0.5 * left_hip + 0.5 * right_hip
        center_bodies = center_body.expand(keypoints.shape[0], 2)
        keypoints = keypoints - center_bodies
        pose_size = self.get_pose_size(keypoints)
        keypoints = keypoints / pose_size
        return keypoints

class PoseClassifier(object):
    def __init__(self, weight_path, class_path, pose_encoder, num_keypoints = 17):
        self.pose_names = self._get_pose_names(class_path)
        self.model = NN_Model(num_keypoints, len(self.pose_names))
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cuda:0")))
        self.model.eval()
        x = torch.rand((1, 17, 2))
        traced_model = torch.jit.trace(self.model, x)
        traced_model.save("./NN_Classifier.jit")
        self.pose_encoder = pose_encoder

    def predict(self, pose_estimation_result):
        keypoints = pose_estimation_result['keypoints'] # Tensor
        inputs = self.pose_encoder.encode(keypoints)
        inputs = inputs.unsqueeze(0)
        pred = self.model(inputs)
        pred = torch.argmax(pred, dim = 1)
        return self.pose_names[pred]

    def _get_pose_names(self, file_paths):
        with open(file_paths, "r") as f:
            pose_names = f.readlines()
            pose_names = [name.strip() for name in pose_names]
        return pose_names
