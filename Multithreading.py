# Multi-threading with OpenCV-Python

from datetime import datetime

import cv2


# Video processing is a computationally intensive task
# I/O operations tend to be a major bottleneck
# Splitting the computational load between multiple threads
# In a single-threaded video processing app, we have the main thread excutes
# the following tasks in an infinitely while loop:
#   get a frame with cv2.VideoCapture.read()
#   process the frame as we need
#   display the processed frame on the screen with a call to cv2.imshow()
# Measure iterations of the main while loop executing per second:

class CountsPerSec:
    """Track the number of occurences (counts) of an arbitrary event and
    returns the freq in occurences (counts) oer second"""

    def __init__(self):
        self._start_time = None
        self._num_occurences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurences/elapsed_time

CountsPerSec = CountsPerSec()

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, 450), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    return frame


def noThreading(source):
    cap = cv2.VideoCapture(source)
    cps = CountsPerSec.start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord('q'):
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

# noThreading()

# Separating thread for getting video frames

from threading import Thread


class VideoGet:
    """
    Class that continuosly gets frames from a VideoCapture object with a dedicated thread.
    """

    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()

    def stop(self):
        self.stopped = True


def threadVideoGet(src):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames
    :param src:
    :return:
    """

    video_getter = VideoGet(src).start()
    cps = CountsPerSec.start()

    while True:
        if cv2.waitKey(1) == ord("q") or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

# threadVideoGet()

class VideoShow:
    """
    Class that continously shows a frame using a dedicated thread
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True


def threadVideoShow(src):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    :param src:
    :return:
    """
    cap = cv2.VideoCapture(src)
    grabbed, frame = cap.read()
    video_show = VideoShow(frame).start()
    cps = CountsPerSec.start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or video_show.stopped:
            video_show.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_show.frame = frame
        cps.increment()

# threadVideoShow()

def threadBoth(src):

    video_getter = VideoGet(src).start()
    video_show = VideoShow(video_getter.frame).start()
    cps = CountsPerSec.start()

    while True:
        if video_getter.stopped or video_show.stopped:
            video_getter.stop()
            video_show.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_show.frame = frame
        # if detect_blur(frame):
        #     print('blur detected')
        cps.increment()


# threadBoth()