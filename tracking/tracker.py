import math
# class 

class Tracker(object):
    def __init__(self, story_num = 6, video_fps = 25):
        self.buffer = []
        self.max_buffer_size = story_num
        self.max_tracking_id = 0
        self.video_fps = video_fps
        self.tracking_period = 3 #for velocity tracking, unit: frame
        self.meter_per_px = 9.0 / 720

    def _box_distance(self, box_a, box_b):
        centerX_a, centerY_a, centerX_b, centerY_b = box_a[0], box_a[1], box_b[0], box_b[1]
        return math.sqrt((centerX_a - centerX_b)*(centerX_a - centerX_b) + (centerY_a - centerY_b)*(centerY_a - centerY_b))

    def update_buffer(self, cur_object_infos):
        if(len(self.buffer) == self.max_buffer_size):
            self.buffer.pop(-1)
        self.buffer.insert(0, cur_object_infos)

    def _estimate_velocity(self, v_prev, z_cur, kalman_gain = 0.5):
        """"
            v_prev: previous estimate
            z_cur: current measurement value
            kalman_gain
            apply this update equation: v_cur = v_prev + kalman_gain * (z_cur - v_prev)
        """
        return v_prev + kalman_gain * (z_cur - v_prev)

    def _measure_velocity(self, object_distance, frame_distance):
        return object_distance * self.video_fps * self.meter_per_px / frame_distance

    def object_tracking(self, cur_object_infos, max_dist_thres = 30):
        """
            Track identity and velocity of object
        """
        # import pdb; pdb.set_trace()
        has_previous = False
        tracking_ids = []
        num_cur_boxes = len(cur_object_infos)
        if(num_cur_boxes == 0):
            self.update_buffer(cur_object_infos)
            return None

        for prev_object_infos in self.buffer:
            if (len(prev_object_infos) != 0):
                has_previous = True

        if not has_previous:
            for i in range(num_cur_boxes):
                tracking_ids.append(self.max_tracking_id)
                cur_object_infos[i].track_id = self.max_tracking_id
                self.max_tracking_id += 1
        else:
            min_dists = [100000]*num_cur_boxes
            # track_ids = [-1] * num_cur_boxes
            track_objects = [None] * num_cur_boxes # store the latest object that having same track_id with object in current step
            for i in range(num_cur_boxes):
                cur_box = cur_object_infos[i].box
                for prev_infos in self.buffer:
                    for prev_info in prev_infos: # boxes: np.array 
                        prev_box = prev_info.box
                        d = self._box_distance(prev_box, cur_box)
                        if(d < min_dists[i]):
                            min_dists[i] = d
                            track_objects[i] = prev_info

            for i in range(num_cur_boxes):
                if(min_dists[i] < max_dist_thres and track_objects[i] is not None):
                    cur_object_infos[i].track_id = track_objects[i].track_id
                    z_n = self._measure_velocity(min_dists[i], cur_object_infos[i].frame_id - track_objects[i].frame_id)
                    # update velocity every 5 frames
                    if (cur_object_infos[i].frame_id % self.tracking_period == 0):
                        # print("Update ", cur_object_infos[i].frame_id)
                        if(track_objects[i].velocity == 0.0): # compare float number (?)
                            cur_object_infos[i].velocity = z_n
                        else:
                            cur_object_infos[i].velocity = self._estimate_velocity(track_objects[i].velocity, z_n)
                        # print("Object ", i, ": ", cur_object_infos[i].velocity)
                    else:
                        # print("Not update")
                        cur_object_infos[i].velocity = track_objects[i].velocity
                        # print("Object ", i, ": ", cur_object_infos[i].velocity)
                    
                else:
                    cur_object_infos[i].track_id = self.max_tracking_id
                    cur_object_infos[i].velocity = 0.0
                    self.max_tracking_id += 1
                tracking_ids.append(cur_object_infos[i].track_id)
        self.update_buffer(cur_object_infos)
        return tracking_ids