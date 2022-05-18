#include "pose_encoder.h"
#include "torch/script.h"
at::Tensor CocoPoseEncoder::calculate_distance(const at::Tensor &a, const at::Tensor &b){
    at::Tensor diff = a - b;
    // std::cout << "diff: " << diff << std::endl;
    at::Tensor dist = torch::sqrt(torch::sum(diff.mul(diff), -1));
    // std::cout << "dist: " << dist << std::endl;
    return dist;
}

float CocoPoseEncoder::get_pose_size(const at::Tensor &keypoints){
    at::Tensor center_shoulder = 0.5 * keypoints[5] + 0.5 * keypoints[6]; // midpoint of right shoulder and left shoulder
    // std::cout << "center_shoulder: " << center_shoulder << std::endl;
    at::Tensor center_body = 0.5 * keypoints[11] + 0.5 * keypoints[12];
    // std::cout << "center_body: " << center_body << std::endl;
    float torso_size = calculate_distance(center_body, center_shoulder).item().to<float>() * 2.5;
    // std::cout << "torso_size: " << torso_size << std::endl;
    at::Tensor center_bodies = center_body.expand_as(keypoints);
    // std::cout << "center_bodies: " << center_bodies << std::endl;
    at::Tensor dist = calculate_distance(center_bodies, keypoints);
    // std::cout << "dist: " << dist << std::endl;
    float max_dist = dist.max().item().to<float>();
    // std::cout << "max_dist: " << max_dist << std::endl;
    float pose_size = (max_dist > torso_size) ? max_dist : torso_size;
    // std::cout << "pose_size: " << pose_size << std::endl;
    return pose_size;
}

void CocoPoseEncoder::encode(const pose_t& pose, at::Tensor& pose_embedding){
    at::Tensor keypoints = torch::empty({NUM_JOINTS , 2});
    auto accessor = keypoints.accessor<float, 2>();
    for (int i = 0; i < pose.keypoints.size(); i++){
        keypoints[i][0] = pose.keypoints[i].x;
        keypoints[i][1] = pose.keypoints[i].y;
    }
    at::Tensor center_body = 0.5 * keypoints[11] + 0.5 * keypoints[12];
    // std::cout << "Center_ body: " << center_body << std::endl;
    at::Tensor center_bodies = center_body.expand_as(keypoints);
    // std::cout << "Center_ bodies: " << center_bodies.sizes() << std::endl;
    keypoints = keypoints - center_bodies;
    // std::cout << "Keypoints: " << keypoints << std::endl;
    float pose_size = get_pose_size(keypoints);
    // std::cout << "pose_size: " << pose_size << std::endl;
    pose_embedding = keypoints / pose_size;
}