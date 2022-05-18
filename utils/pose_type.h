#ifndef POSE_TYPES
#define POSE_TYPES
#include<darknet.h>
#include<vector>
#include <opencv2/core.hpp>
#include "image.h"
#include "yolo_v2_class.hpp"
typedef struct{
    int w;
    int h;
}scale_t;

typedef struct{
    float scaleX;
    float scaleY;
}scale_factor_t;

typedef struct kp_t{
    int x {};
    int y {};
    float prob {};
}kp_t;

typedef struct point_t{
    float x {};
    float y {};
}point_t;

// THAY DOI SO NUM_JOINT O DAY
typedef struct pose_t{
    std::vector<kp_t> keypoints{17}; 
    float score {};
    // bbox_t bounding_box;
    std::string name {};
}pose_t;

typedef struct object_info
{
    bbox_t box;
    pose_t pose;
    float velocity {0};
    int frame_id {0};
} object_info;

typedef struct app_data_item_t
{
    // TODO: create a struct containing bbox, pose and velocity for each object
    cv::Mat cap_frame;
    image detect_image;
    int frame_id {0};
    std::vector<object_info> infos;
    bool exit_flag{false}; // thread will stop when it hits item having exit_flag = true
}app_data_item_t;

#endif


// typedef struct app_data_item_t
// {   
//     // TODO: create a struct containing bbox, pose and velocity for each object
//     cv::Mat cap_frame;
//     image detect_image;
//     std::vector<bbox_t> bboxes;
//     vector<pose_t> poses;
//     bool exit_flag{false}; // thread will stop when it hits item having exit_flag = true
//     vector<float> velocities;
// } app_data_item_t;