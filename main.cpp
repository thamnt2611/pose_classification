#include "darknet.h"
#include "detector.h"
#include "yolo_v2_class.hpp"
#include "pose_estimator.h"
#include <torch/script.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

void show_result(cv::Mat &img, std::vector<bbox_t> &bboxes){
    for (auto box : bboxes){
        int x = box.x;
        int y = box.y;
        int w = box.w;
        int h = box.h;
        int x_max = x + w/2;
        int x_min = x - w/2;
        int y_max = y + h/2;
        int y_min = y - h/2;
        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, 2);
    }

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display detection result", img);
    cv::waitKey(0);
}

void show_pose_result(cv::Mat &img, std::vector<pose_t> &poses){
    for(pose_t pose : poses){
        for (kp_t kp : pose.keypoints){
            int kp_x = kp.x;
            int kp_y = kp.y;
            cv::circle(img, cv::Point {kp_x, kp_y}, 3, cv::Scalar {0, 255, 255}, -1);
        } 
    }
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
    cv::imshow("Display detection result", img);
    cv::waitKey(0);
}

int main(int argc, const char* argv[]){
    char* names_file = "../model/coco_labels.txt";
    char* cfg_file = "../model/yolov3-spp.cfg";
    char* weights_file = "../model/yolov3-spp.weights";
    char* filename = "../input/yoga_pose.jpg";
    char* pose_weight = "../model/FastPose.jit";
    HumanDetector detector {cfg_file, weights_file, names_file};
    std::vector<bbox_t> bboxes = detector.predict(filename);

    cv::Mat img = cv::imread(filename);
    PoseEstimator estimator{pose_weight};
    pose_t p;
    vector<pose_t> poses(bboxes.size(), p);
    estimator.predict(img, bboxes, poses);
    show_pose_result(img, poses);
    return 0;
}