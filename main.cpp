#include "darknet.h"
#include "image.h"
#include "detector.h"
#include "yolo_v2_class.hpp"
#include "pose_estimator.h"
#include "tracker.h"
#include "classifier.h"
#include "pose_encoder.h"
#include "pose_type.h"
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
#include <sstream>

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
            while (!a_ptr.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            ptr.reset(a_ptr.exchange(nullptr));
        } while (!ptr);
        return *ptr;
    }
};

void draw_tracking_result(cv::Mat &img, std::vector<object_info> &infos)
{
    for (auto info : infos)
    {
        bbox_t box = info.box;
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
        std::stringstream v_stream;
        v_stream << std::fixed << std::setprecision(2) << info.velocity;
        std::string v = v_stream.str();
        std::string text = "id: " + std::to_string(info.box.track_id) + "; v: " + v +  " " + info.pose.name;
        cv::putText(img, text, cv::Point{x, y_min}, cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar{255, 0, 255});
    }
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display detection result", img);
    cv::waitKey(25);
}

void draw_pose(cv::Mat &img, std::vector<pose_t> &poses, std::vector<bbox_t> &boxes) // mspf: ms / frame
{
    for (int i = 0; i < poses.size(); i++)
    {
        pose_t pose = poses[i];
        bbox_t box = boxes[i];
        int x = box.x;
        int y = box.y;
        int w = box.w;
        int h = box.h;
        int x_max = x + w / 2;
        int x_min = x - w / 2;
        int y_max = y + h / 2;
        int y_min = y - h / 2;
        for (kp_t kp : pose.keypoints)
        {
            int kp_x = kp.x;
            int kp_y = kp.y;
            cv::circle(img, cv::Point{kp_x, kp_y}, 3, cv::Scalar{0, 255, 255}, -1);
        }
        cv::rectangle(img, cv::Point(x_min, y_min), cv::Point(x_max, y_max), cv::Scalar{0, 255, 255}, 2);
        std::string text = "Man " + std::to_string(box.track_id);
        cv::putText(img, text, cv::Point{x, y_min}, cv::FONT_HERSHEY_SIMPLEX,1, cv::Scalar{255, 0, 255});
    }
}

void process_video(const char *video_path,
                   HumanDetector &detector,
                   Tracker &tracker,
                   PoseEstimator &estimator,
                   PoseClassifier &classifier,
                   bool &main_exit_flag)
{
    cv::VideoCapture cap;
    cap.open(video_path);
    cv::Mat frame;
    app_data_item_t data_item;
    int frame_cnt = 0;
    do{
        // capture image
        cap.read(frame);
        if(frame.empty()){
            std::cout << "EMPTY" << std::endl;
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
            int step = frame_resized.step;
            for (k = 0; k < c; ++k) {
                for (j = 0; j < h; ++j) {
                    for (i = 0; i < w; ++i) {
                        int dst_index = i + w * j + w * h*k;
                        int src_index = j*step + i*c + k;
                        im.data[dst_index] = (float)frame_resized.data[src_index] / 255.;
                    }
                }
            }
            
            // detect image
            data_item.cap_frame = frame;
            data_item.detect_image = im;
            data_item.frame_id = frame_cnt;
            vector<bbox_t> boxes = detector.detect(im, scale_t {data_item.cap_frame.cols, data_item.cap_frame.rows});
            if(boxes.size() == 0) continue;
            object_info tmp;
            vector<object_info> infos (boxes.size(), tmp);
            for (int i = 0; i < boxes.size(); i++){
                infos[i].box = boxes[i];
                infos[i].frame_id = data_item.frame_id;
            }
            data_item.infos = infos;

            // estimate
            pose_t p;
            vector<pose_t> poses(data_item.infos.size(), p);
            vector<bbox_t> bboxes;
            for (auto& info : data_item.infos){
                bboxes.push_back(info.box);
            }
            if(data_item.infos.size() != 0){
                estimator.predict(data_item.cap_frame, bboxes, poses);
            }

            // classify
            std::vector<std::string> pred_names(poses.size(), "");
            classifier.predict(poses, pred_names);
            for (int i = 0; i < data_item.infos.size(); i++){
                poses[i].name = pred_names[i];
                data_item.infos[i].pose = poses[i];
            }

            //display
            tracker.track_object(data_item.infos, 6, 30);
            for (auto& info : data_item.infos){
                std::cout << "Frame: " << info.frame_id << "; Object_id: " << info.box.track_id << "; Velocity: " << info.velocity << "; Pose: " << info.pose.name << std::endl;
            }

            draw_tracking_result(frame, data_item.infos);
            frame_cnt++;
        }
    } while(!(data_item.exit_flag || main_exit_flag));
    cv::destroyAllWindows();
    cap.release();
    std::cout << "t_cap exit \n"; 
    exit_flag2 = true;
}

void process_image(char *image_path, 
                HumanDetector &detector,
                PoseEstimator &estimator,
                PoseClassifier &classifier)
{

    std::cout << "Detect image..." << std::endl;
    vector<bbox_t> boxes = detector.detect(image_path);
    int num = boxes.size();
    pose_t p;
    vector<pose_t> poses(num, p);
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    std::cout << "Pose estimate image..." << std::endl;
    estimator.predict(img, boxes, poses);
    vector<std::string> pred_names(poses.size(), "");
    classifier.predict(poses, pred_names);
    for (auto name : pred_names){
        std::cout << name << std::endl;
    }
    // std::cout << "Display image" << std::endl;
    // draw_pose(img, poses, boxes);
    // cv::imshow("Display detection result", img);
    // cv::waitKey(0);
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
        char* model_path = "../model/NN_Classifier.jit";

        if (argc > 1){
            input_path = argv[1];
        }
        HumanDetector detector{cfg_file, weights_file, names_file};
        PoseEstimator estimator{pose_weight, 17};
        Tracker tracker{};
        CocoPoseEncoder pose_encoder = CocoPoseEncoder();
        vector<std::string> pose_names {"run", "walk", "stand"};
        PoseClassifier classifier = PoseClassifier(model_path, pose_encoder, pose_names);
        char *type = input_path + strlen(input_path) - 3;
        if (strcmp(type, "jpg") == 0)
        {
            process_image(input_path, detector, estimator, classifier);
        }
        else if (strcmp(type, "mp4") == 0)
        {
            process_video(input_path, detector, tracker, estimator, classifier, main_exit_flag);
        }
    }
    std::cout << "Stop" << std::endl;
    return 0;
}