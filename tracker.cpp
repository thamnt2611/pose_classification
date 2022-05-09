#include<tracker.h>
#include<cmath>

int Tracker::calculate_box_distance(const bbox_t &box_a, const bbox_t &box_b){
    int a_x = box_a.x;
    int a_y = box_a.y;
    int b_x = box_b.x;
    int b_y = box_b.y;
    return sqrt((a_x - b_x) * (a_y - b_y) + (a_y - b_y) * (a_y - b_y));
}

void Tracker::update_buffer(std::vector<object_info> &cur_object_infos){
    if(buffer.size() == max_buffer_size)){
        buffer.pop_back();
    }
    buffer.push_front(cur_object_infos);
}

float Tracker::estimate_velocity(float v_prev, float z_cur, float kalman_gain){
    return v_prev + kalman_gain * (z_cur - v_prev);
}

float Tracker::measure_velocity(int object_distance, int frame_distance){
    return object_distance * video_fps * meter_per_px / frame_distance;
}

void Tracker::track_object(std::vector<object_info>& cur_object_infos, int frame_stories, int max_dist){
    // std::cout << "Num boxes: " << cur_boxes.size() << std::endl;
    // std::cout << "Num in queue: " << buffer.size() << std::endl;
    // std::cout << "Cur_box id: " << std::endl;
    if (cur_object_infos.size() == 0){
        update_buffer(cur_object_infos);
        return;
    }

    bool has_prev_object = false;
    for (auto infos : buffer){
        if (infos.size() != 0){
            has_prev_object = true;
            break;
        }
    }

    if (!has_prev_object){
        // NOTE: code ntn thi co thay doi duoc khong?
        for (auto& info : cur_object_infos){
            info.box.track_id = max_tracking_id;
            max_tracking_id++;
        }
    } 
    else{
        std::vector<int> min_dists (cur_object_infos.size(), INT_MAX);
        object_info tmp; 
        tmp.frame_id = -1; // mark empty object infos
        std::vector<object_info> tracked_objects(cur_object_infos.size(), tmp);
        for (auto &infos : buffer){
            for (auto &prev_info : infos){
                for(int i = 0; i < cur_object_infos.size(); i++){
                    int dist = calculate_box_distance(prev_info.box, cur_object_infos[i].box);
                    // std::cout << "DIST: " << dist << std::endl;
                    if (dist < min_dists[i]){
                        min_dists[i] = dist;
                        tracked_objects[i] = prev_info;
                    }
                }
            }
        }

        for(int i = 0; i < cur_object_infos.size(); i++){
            if((min_dists[i] < max_dist) && (tracked_objects[i].frame_id != -1)){
                cur_object_infos[i].box.track_id = tracked_objects[i].box.track_id;
                float z_n = measure_velocity(min_dists[i], cur_object_infos[i].frame_id - tracked_objects[i].frame_id);
                if (cur_object_infos[i].frame_id % tracking_period == 0){
                    if (tracked_objects[i].velocity == 0){ 
                        cur_object_infos[i].velocity = z_n;
                    }
                    else{
                        cur_object_infos[i].velocity = estimate_velocity(tracked_objects[i].velocity, z_n);
                    }
                }
                else{
                    cur_object_infos[i].velocity = tracked_objects[i].velocity;
                }

                // std::cout << "TRACKING: " << i << " " << cur_boxes[i].track_id << std::endl;
            }
            else{
                cur_object_infos[i].box.track_id = max_tracking_id;
                max_tracking_id++;
                cur_object_infos[i].velocity = 0;
                // std::cout << "NEW: " << i << " " << cur_boxes[i].track_id << std::endl;
            }
        }
    }
    update_buffer(cur_object_infos);
}