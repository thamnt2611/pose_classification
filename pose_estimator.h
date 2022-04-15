#include <map>
#include <string>
#include <vector>
#include <torch/script.h>
#include <opencv2/core/types.hpp>
#include "yolo_v2_class.hpp"
#include "pose_type.h"
using namespace std;

class PoseEstimator{
    private:
        map<string, string> cfg;
        torch::jit::script::Module model;

    public:
    // THAY DOI O DAY
        int NUM_JOINTS {26};
        int INPUT_W {192};
        int INPUT_H {256};
        int HEATMAP_W {48};
        int HEATMAP_H {64};
        PoseEstimator(const char* model_path);
        void _load_model(const char* model_path);
        void _preprocess(const cv::Mat& orig_img, 
                        const vector<bbox_t>& boxes, 
                        vector<at::Tensor>& inps, 
                        vector<bbox_t>& n_boxes);

        void _post_process(const at::Tensor& heatmap,
                        const vector<bbox_t>& boxes,
                        vector<pose_t>& poses);

        void predict(const cv::Mat& orig_img, 
                    const vector<bbox_t>& detected_boxes,
                    vector<pose_t>& preds);
        
};