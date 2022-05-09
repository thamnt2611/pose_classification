#include "darknet.h"
#include "image.h"
#include "yolo_v2_class.hpp"
#include <string>
#include <vector>
#include <exception>
#include <signal.h>
#include <stdlib.h>

typedef struct app_data_item_t
{   
    // TODO: create a struct containing bbox, pose and velocity for each object
    cv::Mat cap_frame;
    image detect_image;
    std::vector<bbox_t> bboxes;
    vector<pose_t> poses;
    bool exit_flag{false}; // thread will stop when it hits item having exit_flag = true
    vector<float> velocities;
} app_data_item_t;

typedef struct app_data_item_t_2
{
    // TODO: create a struct containing bbox, pose and velocity for each object
    cv::Mat cap_frame;
    image detect_image;
    std::vector<object_info> infos;
    bool exit_flag{false}; // thread will stop when it hits item having exit_flag = true
}app_data_item_t_2;

typedef struct object_info
{
    bbox_t box;
    pose_t pose;
    float velocity {0};
    int frame_id {0};
} object_info;