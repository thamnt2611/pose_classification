#include "darknet.h"
#include "yolo_v2_class.hpp"
#include<vector>
#include<set>
#include "stdlib.h"
class HumanDetector
{
    private:
        network* net;
        const std::vector<int> DARKNET_INPUT_SHAPE {416, 416, 3};
        const float NMS {0.45};
        const float THRESH {0.5};
        const float HIER_THRESH {0.5};
        const std::set<int> INTEREST_CLASSES {0};

        // TODO: Dinh nghia sau
        std::vector<std::string> LABELS;

        // TODO: load network

    public:
        HumanDetector(char* cfg_file, char* weight_file, char* label_file);
        std::vector<bbox_t> predict(char* image_path);
        void _load_labels(char* label_file);
        image _preprocess_input(image orig_image);
        image _load_detector_image(char* image_path);
        std::vector<bbox_t> _postprocess(detection* dets, int nboxes, std::vector<float>& scale_factor);
};
