#include "darknet.h"
#include "detector.h"
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
}

image HumanDetector::_preprocess_input(image orig_image){
    int W = network_width(net);
    int H = network_height(net);
    int C = 3;
    // image darknet_image = make_image(W, H, C);
    // image rgb_image = cv::cvtColor(orig_image, cv::COLOR_BGR2RGB);
    image darknet_image;
    darknet_image = resize_image(orig_image, W, H);
    // memcpy(darknet_image.data, resized_image.data, darknet_image.w*darknet_image.h*sizeof(float));
    return darknet_image;
}

image HumanDetector::_load_detector_image(char* image_path){
    cv::Mat cv_img = cv::imread(image_path, cv::IMREAD_COLOR);
    image img = load_image(image_path, cv_img.cols, cv_img.rows, 3);
    return img;
}

std::vector<bbox_t> HumanDetector::predict(char* image_path){
    image orig_image = _load_detector_image(image_path);
    image darknet_image = _preprocess_input(orig_image);
    float *prediction = network_predict_image(net, darknet_image);
    int nboxes = 0;
    int letterbox = 0;
    detection *dets = get_network_boxes(net, darknet_image.w, darknet_image.h, THRESH, HIER_THRESH, 0, 0, &nboxes, letterbox);
    do_nms_sort(dets, nboxes, LABELS.size(), NMS);

    int H = orig_image.h; int W = orig_image.w;
    int n_W = network_width(net); int n_H = network_height(net);
    std::vector<float> scale_factor {W*1.0 / n_W, H*1.0 / n_H};
    std::vector<bbox_t> res = _postprocess(dets, nboxes, scale_factor);
    return res;
}  

std::vector<bbox_t> HumanDetector::_postprocess(detection* dets, int nboxes, std::vector<float>& scale_factor){
    float scaleX = scale_factor[0];
    float scaleY = scale_factor[1];
    std::vector<bbox_t> res;
    for(int i = 0; i < nboxes; i++){
        box b = dets[i].bbox;
        int class_idx = max_index(dets[i].prob, LABELS.size());
        float score = dets[i].prob[class_idx];
        if ((INTEREST_CLASSES.find(class_idx) != INTEREST_CLASSES.end()) && (score > 0)){
            bbox_t bbox;
            bbox.x = b.x * scaleX;
            bbox.y = b.y * scaleY;
            bbox.w = b.w * scaleX;
            bbox.h = b.h * scaleY;
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

void HumanDetector::_load_labels(char* label_file){
    std::ifstream lb_file (label_file); 
    std::string line;
    while(std::getline(lb_file, line)){
        LABELS.push_back(line);
    }
    lb_file.close();
}