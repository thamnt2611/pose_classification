import numpy as np 
import cv2
import argparse
import abc
import time
import os
np.random.seed(42)
from ctypes import *

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("best_class_idx", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

class DarknetLib(object):
    def __init__(self):
        if os.name == "posix":
            cwd = os.path.dirname(__file__)
            # import pdb; pdb.set_trace()
            lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
        elif os.name == "nt":
            cwd = os.path.dirname(__file__)
            os.environ['PATH'] = cwd + ';' + os.environ['PATH']
            lib = CDLL("darknet.dll", RTLD_GLOBAL)
        else:
            print("Unsupported OS")
            exit
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        self.copy_image_from_bytes = lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

        self.predict = lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = lib.cuda_set_device
        self.init_cpu = lib.init_cpu

        self.make_image = lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_batch_detections = lib.free_batch_detections
        self.free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        self.free_ptrs = lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.free_network_ptr = lib.free_network_ptr
        self.free_network_ptr.argtypes = [c_void_p]
        self.free_network_ptr.restype = c_void_p

        self.do_nms_obj = lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.predict_image_letterbox = lib.network_predict_image_letterbox
        self.predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        self.predict_image_letterbox.restype = POINTER(c_float)

        self.network_predict_batch = lib.network_predict_batch
        self.network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                        c_float, c_float, POINTER(c_int), c_int, c_int]
        self.network_predict_batch.restype = POINTER(DETNUMPAIR)
        self.lib = lib

    def network_width(self, net):
        return self.lib.network_width(net)


    def network_height(self, net):
        return self.lib.network_height(net)