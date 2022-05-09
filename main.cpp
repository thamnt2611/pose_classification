#include "darknet.h"
#include "image.h"
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
#include <chrono>
#include <thread>
#include <exception>
#include <signal.h>
#include <stdlib.h>
#include <data_format.h>

class InterruptException : public std::exception
{
public:
    InterruptException(int s) : S(s) {}
    int S;
};

void sig_to_exception(int s)
{
    throw InterruptException(s);
}

bool exit_flag = false;
bool exit_flag2 = false;
template <typename T>

class send_one_replaceable_object_t
{
    std::atomic<T *> a_ptr = {nullptr};

public:
    void send(T const &_obj)
    {
        T *new_ptr = new T;
        *new_ptr = _obj;
        // TODO: The `unique_ptr` prevents a scary memory leak, why?
        std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
    }

    T receive()
    {
        std::unique_ptr<T> ptr;
        do
        {
            while (!a_ptr)
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(nullptr));
        } while (!ptr);
        return *ptr;
    }
};

void show_result(cv::Mat &img, std::vector<bbox_t> &bboxes)
{
    for (auto box : bboxes)
    {
        int x = box.x;
        int y = box.y;
        int w = box.w;
        int h = box.h;
        int x_max = x + w / 2;
        int x_min = x - w / 2;
        int y_max = y + h / 2;
        int y_min = y - h / 2;
        cv::Scalar color = cv::Scalar(255, 255, 255);
        cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, 2);
    }

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display detection result", img);
    cv::waitKey(0);
}

void draw_pose(cv::Mat &img, std::vector<pose_t> &poses) // mspf: ms / frame
{
    for (pose_t pose : poses)
    {
        for (kp_t kp : pose.keypoints)
        {
            int kp_x = kp.x;
            int kp_y = kp.y;
            // std::cout << "Keypoints: " << kp_x << "; " << kp_y << std::endl;
            cv::circle(img, cv::Point{kp_x, kp_y}, 3, cv::Scalar{0, 255, 255}, -1);
            bbox_t box = pose.bounding_box;
            int x = box.x;
            int y = box.y;
            int w = box.w;
            int h = box.h;
            int x_max = x + w / 2;
            int x_min = x - w / 2;
            int y_max = y + h / 2;
            int y_min = y - h / 2;
            cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar{0, 255, 255}, 2);
            std::string text = "Man " + std::to_string(box.track_id);
            cv::putText(img, text, cv::Point{x, y_min}, cv::FONT_HERSHEY_SIMPLEX,1, cv::Scalar{255, 0, 255});
        }
    }
}

void process_video(const char *video_path,
                   HumanDetector &detector,
                   PoseEstimator &estimator,
                   bool &main_exit_flag)
{
    std::thread t_cap, t_det, t_estimate;
    send_one_replaceable_object_t<app_data_item_t> cap2det, det2pose, pose2display;

    t_cap = std::thread([&]()
                        {
        cv::VideoCapture cap;
        cap.open(video_path);
        cv::Mat frame;
        app_data_item_t data_item;
        do{
            cap >> frame;
            if(frame.empty()){
                data_item.exit_flag = true;
            }
            else{
                cv::Mat frame_rgb;
                cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
                cv::Mat frame_resized;
                cv::resize(frame_rgb, frame_resized, 
                        cv::Size {detector.get_input_width(),
                        detector.get_input_height()});
                image im = make_image(frame_resized.cols, frame_resized.rows, 3);
                int i, k, j;
                int w = im.w;
                int h = im.h;
                int c = im.c;
                for (k = 0; k < c; ++k) {
                    for (j = 0; j < h; ++j) {
                        for (i = 0; i < w; ++i) {
                            int dst_index = i + w * j + w * h*k;
                            int src_index = k + c * i + c * w*j;
                            im.data[dst_index] = (float)frame_resized.data[src_index] / 255.;
                        }
                    }
                }
                
                data_item.cap_frame = frame;
                data_item.detect_image = im;
                cap2det.send(data_item);
            }
        } while(!(data_item.exit_flag || main_exit_flag));
        cv::destroyAllWindows();
        cap.release();
        std::cout << "t_cap exit \n"; });

    t_det = std::thread([&]()
                        {
        app_data_item_t data_item;
        do{
            data_item = cap2det.receive();
            if(data_item.exit_flag){
                det2pose.send(data_item);
                break;
            }
            image img = data_item.detect_image;
            vector<bbox_t> boxes = detector.detect(img, scale_t {data_item.cap_frame.cols, data_item.cap_frame.rows});
            // std::cout << "Box size: " << boxes.size() << std::endl;
            data_item.bboxes = boxes;
            det2pose.send(data_item);
        } while(!main_exit_flag);
        std::cout << "t_det exit \n"; });

    t_estimate = std::thread([&]()
                             {
        app_data_item_t data_item;
        do{ 
            data_item = det2pose.receive();
            if(data_item.exit_flag){
                pose2display.send(data_item);
                break;
            }
            pose_t p;
            vector<pose_t> poses(data_item.bboxes.size(), p);
            if(data_item.bboxes.size() != 0){
                estimator.predict(data_item.cap_frame, data_item.bboxes, poses);
            }
            data_item.poses = poses;
            pose2display.send(data_item);
            
        } while (!main_exit_flag);
        std::cout << "t_estimate exit \n"; });

    app_data_item_t data_item;
    while (true)
    {
        data_item = pose2display.receive();
        if (data_item.exit_flag)
        {
            break;
        }
        draw_pose(data_item.cap_frame, data_item.poses);
        cv::imshow("Display detection result", data_item.cap_frame);
        char c = (char)cv::waitKey(25);
        if (c == 27)
        {
            main_exit_flag = true;
            break;
        }
    }
    std::cout << "app exit ...\n";
    std::cout << "ok \n";
    if (t_cap.joinable())
        t_cap.join();
    if (t_det.joinable())
        t_det.join();
    if (t_estimate.joinable())
        t_estimate.join();

    exit_flag2 = true;
}

void process_image(char *image_path, HumanDetector &detector,
                   PoseEstimator &estimator)
{

    std::cout << "Detect image..." << std::endl;
    vector<bbox_t> boxes = detector.detect(image_path);
    pose_t p;
    vector<pose_t> poses(boxes.size(), p);
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    std::cout << "Pose estimate image..." << std::endl;
    estimator.predict(img, boxes, poses);

    std::cout << "Display image" << std::endl;
    draw_pose(img, poses);
    cv::imshow("Display detection result", img);
    cv::waitKey(0);
}

int main(int argc, char *argv[])
{
    bool main_exit_flag = false;
    // try
    {
        char *names_file = "../model/coco_labels.txt";
        char *cfg_file = "../model/yolov3-spp.cfg";
        char *weights_file = "../model/yolov3-spp.weights";
        char *input_path = "../input/bridge.mp4";
        char *pose_weight = "../model/FastPose.jit";

        if (argc > 1){
            input_path = argv[1];
        }
        HumanDetector detector{cfg_file, weights_file, names_file};
        PoseEstimator estimator{pose_weight};
        char *type = input_path + strlen(input_path) - 3;
        if (strcmp(type, "jpg") == 0)
        {
            process_image(input_path, detector, estimator);
        }
        else if (strcmp(type, "mp4") == 0)
        {
            process_video(input_path, detector, estimator, main_exit_flag);
        }
    }
    // catch (InterruptException &e)
    // {
    //     std::cout << "Ctrl C" << std::endl;
    //     main_exit_flag = true;
    // }
    // while (!exit_flag2)
    //     ;
    std::cout << "Stop" << std::endl;
    return 0;
}