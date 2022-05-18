#ifndef POSE_CLASSIFIER
#define POSE_CLASSIFIER
#include "pose_type.h"
#include "pose_encoder.h"
#include <torch/script.h>
#include <vector>
class PoseClassifier{
    private:
        torch::jit::script::Module model;
        CocoPoseEncoder encoder;
        std::vector<std::string> class_names;

    public:
        PoseClassifier(char* model_path, const CocoPoseEncoder& m_encoder, const std::vector<std::string>& m_class_names);
        void predict(const std::vector<pose_t> &poses, std::vector<std::string> &pred);
};
#endif