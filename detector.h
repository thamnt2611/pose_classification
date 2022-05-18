#include "darknet.h"
#include "yolo_v2_class.hpp"
#include<vector>
#include<set>
#include "stdlib.h"
#include "pose_type.h"
class HumanDetector
{
    private:
        network* net;
        const float NMS {0.45};
        const float THRESH {0.5};
        const float HIER_THRESH {0.5};
        const std::set<int> INTEREST_CLASSES {0};
        int INPUT_WIDTH;
        int INPUT_HEIGHT;
        // TODO: Dinh nghia sau
        std::vector<std::string> LABELS;
        int max_track_id {0};
        std::deque<std::vector<bbox_t> > prev_bboxes_deque;

    public:
        HumanDetector(char* cfg_file, char* weight_file, char* label_file);
        std::vector<bbox_t> detect(char* image_path);
        std::vector<bbox_t> detect(image darknet_image, scale_t target_scale);
        void _load_labels(char* label_file);
        image _preprocess_input(image orig_image);
        image _load_detector_image(char* image_path);
        std::vector<bbox_t> get_bbox(detection* dets, int nboxes);
        void scale_bbox(std::vector<bbox_t>& n_boxes, const std::vector<bbox_t>& o_boxes, scale_factor_t& scale_factor);
        int get_input_width() const;
        int get_input_height() const;
        int get_then_update_max_track_id();
        void update_boxes_track_id(std::vector<bbox_t>& boxes, int frame_stories, int max_dist);
        int frame_num {0};
};
