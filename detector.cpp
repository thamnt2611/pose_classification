#include "darknet.h"
#include "detector.h"
#include "pose_type.h"
// #include "utils.h"
#include <set>
#include "yolo_v2_class.hpp"
#include "image.h"
#include "opencv2/core/core.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <fstream>

int max_index(float* arr, int num){
    int max_idx = 0;
    for(int i = 0; i < num; i++){
        if(arr[i] > arr[max_idx]){
            max_idx = i;
        }
    }
    return max_idx;
}

HumanDetector::HumanDetector(char* cfg_file, char* weight_file, char* label_file){
    net = load_network_custom(cfg_file, weight_file, 0, 1);
    _load_labels(label_file);
    INPUT_WIDTH = network_width(net);
    INPUT_HEIGHT = network_height(net);
}

image HumanDetector::_preprocess_input(image orig_image){
    int C = 3;
    // image darknet_image = make_image(W, H, C);
    // image rgb_image = cv::cvtColor(orig_image, cv::COLOR_BGR2RGB);
    image darknet_image;
    darknet_image = resize_image(orig_image, get_input_width(), get_input_height());
    // memcpy(darknet_image.data, resized_image.data, darknet_image.w*darknet_image.h*sizeof(float));
    return darknet_image;
}

image HumanDetector::_load_detector_image(char* image_path){
    cv::Mat cv_img = cv::imread(image_path, cv::IMREAD_COLOR);
    image img = load_image(image_path, cv_img.cols, cv_img.rows, 3);
    return img;
}

std::vector<bbox_t> HumanDetector::detect(char* image_path){
    image orig_image = _load_detector_image(image_path);
    image darknet_image = _preprocess_input(orig_image);
    float *prediction = network_predict_image(net, darknet_image);
    int nboxes = 0;
    int letterbox = 0;
    detection *dets = get_network_boxes(net, darknet_image.w, darknet_image.h, THRESH, HIER_THRESH, 0, 0, &nboxes, letterbox);
    do_nms_sort(dets, nboxes, LABELS.size(), NMS);

    std::vector<bbox_t> boxes = get_bbox(dets, nboxes);
    std::vector<bbox_t> res_boxes;
    scale_factor_t scale_factor {orig_image.w*1.0 / get_input_width(), orig_image.h*1.0 / get_input_height()};
    scale_bbox(res_boxes, boxes, scale_factor);
    return res_boxes;
}  

std::vector<bbox_t> HumanDetector::detect(image darknet_image, scale_t target_scale){
    float *prediction = network_predict_image(net, darknet_image);
    int nboxes = 0;
    int letterbox = 0;
    detection *dets = get_network_boxes(net, darknet_image.w, darknet_image.h, THRESH, HIER_THRESH, 0, 0, &nboxes, letterbox);
    do_nms_sort(dets, nboxes, LABELS.size(), NMS);
    std::vector<bbox_t> boxes = get_bbox(dets, nboxes);
    std::vector<bbox_t> res_boxes;
    scale_factor_t scale_factor {target_scale.w*1.0 / get_input_width(), target_scale.h*1.0 / get_input_height()};
    scale_bbox(res_boxes, boxes, scale_factor);
    // update_boxes_track_id(res_boxes, 5, 50);
    return res_boxes;
}

void HumanDetector::scale_bbox(std::vector<bbox_t>& n_boxes, const std::vector<bbox_t>& o_boxes, scale_factor_t& scale_factor){
    float scaleX = scale_factor.scaleX;
    float scaleY = scale_factor.scaleY;
    bbox_t bbox;
    for (auto b : o_boxes){
        bbox.x = b.x * scaleX;
        bbox.y = b.y * scaleY;
        bbox.w = b.w * scaleX;
        bbox.h = b.h * scaleY;
        bbox.obj_id = b.obj_id;
        bbox.prob = b.prob;
        bbox.track_id = b.track_id;
        bbox.frames_counter = b.frames_counter;
        bbox.x_3d = b.x_3d;
        bbox.y_3d = b.y_3d;
        bbox.z_3d = b.y_3d;
        n_boxes.push_back(bbox);
    }
}

std::vector<bbox_t> HumanDetector::get_bbox(detection* dets, int nboxes){
    std::vector<bbox_t> res;
    for(int i = 0; i < nboxes; i++){
        box b = dets[i].bbox;
        int class_idx = max_index(dets[i].prob, LABELS.size());
        float score = dets[i].prob[class_idx];
        if ((INTEREST_CLASSES.find(class_idx) != INTEREST_CLASSES.end()) && (score > 0)){
            bbox_t bbox;
            bbox.x = b.x;
            bbox.y = b.y;
            bbox.w = b.w;
            bbox.h = b.h;
            bbox.obj_id = class_idx;
            bbox.prob = score;
            bbox.track_id = 0;
            bbox.frames_counter = 0;
            bbox.x_3d = NAN;
            bbox.y_3d = NAN;
            bbox.z_3d = NAN;
            res.push_back(bbox);
        }
    }
    return res;
}

int HumanDetector::get_input_height() const{
    return INPUT_HEIGHT;
}

int HumanDetector::get_input_width() const{
    return INPUT_WIDTH;
}

void HumanDetector::_load_labels(char* label_file){
    std::ifstream lb_file (label_file); 
    std::string line;
    while(std::getline(lb_file, line)){
        LABELS.push_back(line);
    }
    lb_file.close();
}

int HumanDetector::get_then_update_max_track_id(){
    return max_track_id++;
}

// void HumanDetector::update_boxes_track_id(std::vector<bbox_t>& cur_boxes, int frame_story, int max_dist){
//     // std::cout << "Num boxes: " << cur_boxes.size() << std::endl;
//     // std::cout << "Num in queue: " << prev_bboxes_deque.size() << std::endl;
//     // std::cout << "Cur_box id: " << std::endl;
//     if (cur_boxes.size() == 0){
//         prev_bboxes_deque.push_front(cur_boxes);
//         if(prev_bboxes_deque.size() > frame_story){
//             prev_bboxes_deque.pop_back();
//         }
//         return;
//     }

//     bool has_prev_boxes = false;
//     for (auto boxes : prev_bboxes_deque){
//         if (boxes.size() != 0){
//             has_prev_boxes = true;
//             break;
//         }
//     }

//     if (!has_prev_boxes){
//         // NOTE: code ntn thi co thay doi duoc khong?
//         for (auto& box : cur_boxes){
//             box.track_id = get_then_update_max_track_id();
//             // std::cout << "NEW: " << box.track_id << std::endl;
//         }
//     } else{
//         std::vector<int> min_box_dist (cur_boxes.size(), INT_MAX);
//         std::vector<int> tmp_id (cur_boxes.size(), 0);
//         for (auto &boxes : prev_bboxes_deque){
//             for (auto &prev_box : boxes){
//                 for(int i = 0; i < cur_boxes.size(); i++){
//                     int dist = sqrt((prev_box.x - cur_boxes[i].x) * (prev_box.x - cur_boxes[i].x)
//                                         + (prev_box.y - cur_boxes[i].y) * (prev_box.y - cur_boxes[i].y));
//                     // std::cout << "DIST: " << dist << std::endl;
//                     if (dist < min_box_dist[i]){
//                         tmp_id[i] = prev_box.track_id;
//                         min_box_dist[i] = dist;
//                     }
//                 }
//             }
//         }

//         for(int i = 0; i < cur_boxes.size(); i++){
//             if(min_box_dist[i] < max_dist){
//                 cur_boxes[i].track_id = tmp_id[i];
//                 // std::cout << "TRACKING: " << i << " " << cur_boxes[i].track_id << std::endl;
//             }
//             else{
//                 cur_boxes[i].track_id = get_then_update_max_track_id();
//                 // std::cout << "NEW: " << i << " " << cur_boxes[i].track_id << std::endl;
//             }
//         }
//     }
//     prev_bboxes_deque.push_front(cur_boxes);
//     if(prev_bboxes_deque.size() > frame_story){
//         prev_bboxes_deque.pop_back();
//     }
// }