
from queue import Queue
import cv2
import sys
from threading import Thread

class ImageQueue(object):
    def __init__(self, input_path, queue_size = 10):
        self.size = queue_size
        self.queue = Queue(queue_size)
        self.ips = 0 # images per second
        self.input_path = input_path

    def start_read_input(self):
        p = Thread(target=self._append_input, args=())
        p.start()

    def _append_input(self):
        if (self.input_path[-3:] == "jpg"):
            self._load_image(self.input_path)
        elif self.input_path[-3:] == "mp4":
            self._load_video(self.input_path)

    def _load_image(self, input_path):
        img = cv2.imread(input_path)
        self.queue.put(img)
    
    def _load_video(self, input_path):
        stream = cv2.VideoCapture(input_path)
        self.ips = stream.get(cv2.CAP_PROP_FPS)
        while (True):
            grabbed, frame = stream.read()
            if not grabbed:
                print(grabbed)
                sys.stdout.flush()
                stream.release()
                break
            else:
                self.queue.put(frame)
        stream.release()

    def get(self):
        return self.queue.get()
