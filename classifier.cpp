#include "classifier.h"
#include <vector>

PoseClassifier::PoseClassifier(char* model_path, const CocoPoseEncoder& m_encoder, const std::vector<std::string>& m_class_names){
    const torch::Device device = torch::Device(torch::kCUDA, 0);
    model = torch::jit::load(model_path, device);
    encoder = m_encoder;
    class_names = m_class_names;
}

void PoseClassifier::predict(const std::vector<pose_t> &poses, std::vector<std::string> &pred){
    at::Tensor kp = torch::empty({encoder.NUM_JOINTS, 2});
    std::vector<at::Tensor> keypoints (poses.size(), kp);
    for (int i = 0; i < poses.size(); i++){
        encoder.encode(poses[i], keypoints[i]);
        keypoints[i] = keypoints[i].unsqueeze(0);
    }
    at::Tensor inputs = torch::cat(keypoints);
    const torch::Device device = torch::Device(torch::kCUDA, 0);
    inputs = inputs.to(device);
    std::vector<torch::jit::IValue> input_batch;
    input_batch.push_back(inputs);
    at::Tensor pred_logit = model.forward(input_batch).toTensor();
    at::Tensor pred_id = pred_logit.argmax(1);
    for (int i = 0; i < pred_id.sizes()[0]; i++){
        int idx = pred_id[i].item<int>();
        pred[i] = class_names[idx];
    }
}
