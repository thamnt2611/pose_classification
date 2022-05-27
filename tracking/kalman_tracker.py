from data_format import ObjectState
from tracking.kalman_filter import KalmanFilter
from cython_bbox import bbox_overlaps as bbox_ious
import numpy as np
import lap

class KalmanFilterTracker(object):
    """
        Box distance: iou distance
        Assign identity for object: Jonker-Volgenant Algorithm for Linear Assignment Problem (module: lab)
        Location and Velocity tracking: Kalman Filter
    """
    def __init__(self):
        self.kalman_filter = KalmanFilter()
        self.buffer = [] # NOTE: buffer: list of object data
        self.frame_id = 0
        self.max_track_id = 0
        self.max_time_lost = 4
        self.kalman_filter = KalmanFilter()

    def update_buffer(self, cur_object_infos):
        self.buffer = [info for info in self.buffer if (self.frame_id - info.frame_id < self.max_time_lost)]
        for info in cur_object_infos:
            self.buffer.append(info)

    def iou_distance(self, a_object_infos, b_object_infos):
        a_tlbrs = [np.array(info.tlbr)for info in a_object_infos]
        b_tlbrs = [np.array(info.tlbr) for info in b_object_infos]
        dist = np.zeros((len(a_tlbrs), len(b_tlbrs)), dtype=np.float)
        if (dist.size == 0):
            return dist
        ious = bbox_ious(
            np.ascontiguousarray(a_tlbrs, dtype=np.float), 
            np.ascontiguousarray(b_tlbrs, dtype=np.float)
        )
        # import pdb; pdb.set_trace()
        dist = 1 - ious
        return dist

    def joint_infos(self, a_infos, b_infos):
        track_ids = set()
        infos = []
        for info in a_infos:
            track_ids.add(info.track_id)
            infos.append(info)
        for info in b_infos:
            if(info.track_id not in track_ids):
                infos.append(info)
        return infos
        
    def linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        matches, unmatched_a, unmatched_b = [], [], []
        cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatched_a, unmatched_b

    def track_object(self, cur_object_infos):
        self.frame_id += 1
        tracked_prev_infos = []
        for prev_info in self.buffer:
            if(prev_info.state == ObjectState.Tracked or prev_info.state == ObjectState.New or prev_info.state == ObjectState.Lost):
                tracked_prev_infos.append(prev_info)
        dist_matrix = self.iou_distance(tracked_prev_infos, cur_object_infos)
        matches, unmatched_tracks, unmatched_dets = self.linear_assignment(dist_matrix, 0.5)
        # print("Frame: ", self.frame_id)
        # print("Buffer before: ")
        # for info in tracked_prev_infos:
        #     print(info.box, info.track_id, info.state)

        for prev_idx, cur_idx in matches:
            cur_info = cur_object_infos[cur_idx]
            prev_info = tracked_prev_infos[prev_idx]
            prev_info.extrapolate()
            if (prev_info.state == ObjectState.Tracked):
                prev_info.update(self.frame_id, cur_info)
            elif (prev_info.state == ObjectState.New):
                prev_info.update(self.frame_id, cur_info)
            elif (prev_info.state == ObjectState.Lost):
                prev_info.re_activate(self.frame_id, cur_info)
            
            # UPDATE current info
            cur_object_infos[cur_idx] = prev_info
            
        # print("Buffer After: ")
        # for info in tracked_prev_infos:
        #     print(info.box, info.track_id, info.state)
        
        for idx in unmatched_tracks:
            track_info = tracked_prev_infos[idx]
            if (track_info.state == ObjectState.Tracked or track_info.state == ObjectState.New):
                track_info.state = ObjectState.Lost
            elif (track_info.state == ObjectState.Lost) and (self.frame_id - track_info.frame_id > self.max_time_lost):
                track_info.state = ObjectState.Removed
        # print("Detected: ")
        for idx in unmatched_dets:
            det = cur_object_infos[idx]
            det.activate(self.kalman_filter, self.frame_id, self.max_track_id)
            self.max_track_id += 1
            # print(det.box, det.track_id, det.state)

        # print("Removed: ")
        # for info in tracked_prev_infos:
        #     if(info.state == ObjectState.Removed):
        #         print(info.box, info.track_id, info.state)

        tracked_prev_infos = [info for info in tracked_prev_infos if info.state != ObjectState.Removed]
        self.buffer = self.joint_infos(tracked_prev_infos, cur_object_infos)

        # print("Buffer Last: ")
        # for info in self.buffer:
        #     print(info.box, info.track_id, info.state)
        # print("-------------")