"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Autonomous vehicle perception system [Beta]
"""
from imutils.video import FileVideoStream
from object_recognition import YOLOV3
from lane_detection import Road
from utility import UI
import collections
import threading
import argparse
import cv2
import os


class Perception:
    def __init__(self, video_path: str, model_path: str):
        # Get current directory and define correct paths
        path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(path, model_path)
        self.video_path = os.path.join(path, video_path)

        # Start videostream on a separate thread
        self.fvs = FileVideoStream(self.video_path).start()

        # Create hash table to keep track of parameters and metrics
        self.mem = collections.defaultdict(int)

        # Set wait time for CV2's waitkey method
        self.mem["time"] = 10

        # Set confidence and threshold for YOLO model
        self.mem["confidence"] = 0.88
        self.mem["threshold"] = 0.3

        # Set default video mode
        self.mem["display_mode"] = "Default"

    def update_fps(self):
        """ Check number of frames processed every second

        Args:
            None

        Returns:
            None
        """
        t = threading.Timer(1.0, self.update_fps)

        # Mark thread as daemon and start it
        t.daemon = True
        t.start()

        self.mem["fps"] = self.mem["cnt"] - self.mem["last_cnt"]
        self.mem["last_cnt"] = self.mem["cnt"]

    def start(self):
        """ Start the perception algorithm

        Args:
            None

        Returns:
            None
        """
        self.update_fps()
        dashboard = UI()
        model = YOLOV3(self.model_path)
        lanes = Road()
        while self.fvs.more():
            # Record the current number of frames in the queue
            self.mem["q-size"] = self.fvs.Q.qsize()

            frame = self.fvs.read()
            if frame is not None:
                # Check display mode
                if self.mem["display_mode"] == "Default":
                    # Get frame's height and width when initialized
                    if self.mem["cnt"] == 0:
                        self.mem["height"], self.mem["width"] = frame.shape[:2]

                    # Run object detection every 5th frame
                    elif self.mem["cnt"] % 5 == 0:
                        self.mem = model.predict(frame, self.mem)

                    # Display objects after obtaining predictions
                    elif self.mem["cnt"] > 5:
                        frame = model.display(frame, self.mem)
                    frame = lanes.detect(frame)

                # Switch to cannny ...
                elif self.mem["display_mode"] == "Canny":
                    frame = lanes.preprocess(frame)

                # ... or the region of interest view
                elif self.mem["display_mode"] == "ROI":
                    frame = lanes.roi(frame)

                # Show user interface
                frame = dashboard.display(frame, self.mem)

                # Display videostream
                cv2.imshow("{}".format(self.video_path.split('/')[-1]), frame)

                # Switch between different display modes
                key = cv2.waitKey(self.mem["time"] - 9)
                if key == 27:
                    break
                elif key == 114:
                    self.mem["display_mode"] = "Canny"
                elif key == 105:
                    self.mem["display_mode"] = "ROI"
                elif key == 100:
                    self.mem["display_mode"] = "Video"
                elif key == 99:
                    self.mem["display_mode"] = "Default"
                elif key == 61:
                    self.mem["time"] -= 5 if self.mem["time"] > 10 else 0
                elif key == 45:
                    self.mem["time"] += 5 if self.mem["time"] < 99 else 0
                self.mem["cnt"] += 1
            else:
                break
        self.stop()

    def stop(self):
        """ Halt all opened windows and the videostream thread

        Args:
            None

        Returns:
            None
        """
        cv2.destroyAllWindows()
        self.fvs.stop()


if __name__ == "__main__":
    # Setup argument parsing
    parser = argparse.ArgumentParser()
    v = "relative path to the video file, e.g. \"resources/sample.mp4\""
    m = "relative path to the model's folder, e.g. \"model\""
    parser.add_argument("-v", "--video_path", type=str, help=v, required=True)
    parser.add_argument("-m", "--model_path", type=str, help=m, required=True)
    args = parser.parse_args()

    system = Perception(args.video_path, args.model_path)
    system.start()
