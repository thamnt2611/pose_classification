#ifndef COCO_POSE_ENCODER
#define COCO_POSE_ENCODER
#include "pose_type.h"
#include <torch/script.h>
class CocoPoseEncoder{
    private:
        at::Tensor calculate_distance(const at::Tensor &a, const at::Tensor &b);
        float get_pose_size(const at::Tensor &keypoints);
    public:
        int NUM_JOINTS = 17;
        void encode(const pose_t& poses, at::Tensor &pose_embedding);
};
#endif