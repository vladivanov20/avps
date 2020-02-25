# Autonomous vehicle perception system [Beta]

![preview](resources/preview.gif)

Features:

* Object detection on the state-of-the-art darknet YOLOv3 model
* Relative distance approximation
* Lane detection using Hough transform
* Feature-rich user interface with multiple display modes, customizable playback speed, and relevant metrics
* More in the future!

Dependencies:

- NumPy - Scientific computing made easy
- OpenCV - Highly optimized real-time computer vision
- Imutils - Multithreaded videostream processing and many more other convenient functions

Installation

```bash
$ pip install numpy
$ pip install opencv-python
$ pip install imutils
```

Usage 

```bash
$ git clone  https://github.com/vladivanov20/avps.git
$ cd avps
$ python3 main.py -v "resources/sample.mp4" -m "model"
```







