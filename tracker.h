#include<pose_type.h>
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
        float meter_per_px {1};

        int calculate_box_distance(const bbox_t &box_a, const bbox_t &box_b);
        void update_buffer(std::vector<object_info> &cur_object_infos);
        float estimate_velocity(float v_prev, float z_cur, float kalman_gain = 0.5);
        float measure_velocity(int object_distance, int frame_distance);
        
    public:
        Tracker(int m_max_buffer_size = 6, int m_video_fps = 25, int m_tracking_period = 3);
        void track_object(std::vector<object_info>& cur_object_infos, int frame_stories, int max_dist = 30);
};      