#include<data_format.h>
#include "yolo_v2_class.hpp"
#include<deque>
#include<vector>

class Tracker
{
    private:
        std::deque<std::vector<object_info> > buffer;
        int max_buffer_size;
        int max_tracking_id {0};
        int video_fps;
        int tracking_period; //for velocity tracking, unit = frame

        int calculate_box_distance(bbox_t box_a, bbox_t box_b);
        void update_buffer(vector<object_info> cur_object_infos);
        void estimate_velocity(float v_prev, float z_cur, float kalman_gain = 0.5);
        float measure_velocity(int object_distance, int frame_distance);
        
    public:
        void track_object(std::vector<object_info>& cur_object_infos, int frame_stories, int max_dist);
}