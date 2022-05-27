import numpy as np
class ObjectState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class ObjectInfo(object): # similar to STrack in ByteTrack
    """
        Include state information of an object in a particular frame
    """
    def __init__(self):
        self.box = None # list 4 item
        self.track_id = -1 # int
        self.pose = None
        self.confidence = 0.0 # float
        self.class_id = -1 
        self.frame_id = 0
        self.pose_name = ""
        self.state = ObjectState.New
     
        self.mean, self.covariance = None, None
        self.kalman_filter = None

    def activate(self, kalman_filter, frame_id, track_id):
        """
            set initial state value for object in the beginning of tracking process
        """
        self.frame_id = frame_id
        self.track_id = track_id
        self.kalman_filter = kalman_filter
        self.mean, self.covariance = self.kalman_filter.initiate(np.array(self.xyah))


    def update(self, frame_id, new_info):
        """
            update state for object having been tracked, basing on new information collected in current frame
        """
        self.frame_id = frame_id
        self.state = ObjectState.Tracked
        self.box = new_info.box
        self.confidence = new_info.confidence
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_info.xyah)

    def re_activate(self, frame_id, new_info):
        """
            update state for object appear again after a few frames
        """
        self.frame_id = frame_id
        self.state = ObjectState.Tracked
        self.box = new_info.box
        self.confidence = new_info.confidence
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_info.xyah)
    
    def extrapolate(self):
        """
            extrapolate state of object, prior to collecting new information in current frame
        """
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
                                                                                            
    @property
    def tlbr(self):
        x, y, w, h = self.box[0], self.box[1], self.box[2], self.box[3]
        xmin = x - (w / 2)
        xmax = x + (w / 2)
        ymin = y - (h / 2)
        ymax = y + (h / 2)
        return [xmin, ymin, xmax, ymax]

    @property
    def xyah(self):
        x, y, w, h = self.box[0], self.box[1], self.box[2], self.box[3]
        # xmin = x - (w / 2)
        # ymin = y - (h / 2)
        a = float(w/h)
        return [x, y, a, h]

    @property
    def corrected_box(self):
        if self.mean is None:
            return self.box
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        return ret

    @property
    def velocity(self):
        return abs(self.mean[4])
        
class FrameData(object):
    def __init__(self):
        self.frame = None
        self.object_infos = [] # list of ObjectData
