class ObjectData(object):
    def __init__(self):
        self.box = None # list 4 item
        self.track_id = -1 # int
        self.pose = None
        self.velocity = -1.0 # float
        self.confidence = 0.0 # float
        self.class_id = -1 
        self.frame_id = 0

class FrameData(object):
    def __init__(self):
        self.frame = None
        self.object_infos = [] # list of ObjectData
