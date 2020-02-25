"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Object Recognition with darknet YOLOv3 model
"""
import numpy as np
import time
import cv2
import os


class YOLOV3:
    def __init__(self, resources_path: str):
        classesPath = os.path.join(resources_path, "coco.names")
        weightsPath = os.path.join(resources_path, "yolov3-320.weights")
        configPath = os.path.join(resources_path, "yolov3-320.cfg")
        np.random.seed(0)

        # Configure class names and colors
        self.classes = [line.rstrip('\n') for line in open(classesPath)]
        defined_colors = np.array([[0, 255, 0], [255, 0, 0], [0, 0, 255]])
        random_colors = np.random.randint(255, size=(len(self.classes) - 3, 3))
        self.colors = np.concatenate([defined_colors, random_colors])

        # Configure model
        self.model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = self.model.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]

    def display(self, frame: np.ndarray, mem: dict) -> np.ndarray:
        """Display detected objects

        Args:
            frame: Native frame
            mem: Hash table of latest parameters and metrics

        Returns:
            frame: Frame with an overlay of the detected objects
        """
        if len(list(mem["indices"])) > 0:
            for i in mem["indices"].flatten():
                object_class = mem["classes"][i]
                box = mem["boxes"][i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                # Calculate relative distance to the object
                relative_distance = 6 - ((h / mem["height"]) * 100) / 6.5
                color = [int(c) for c in self.colors[object_class]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Only display statistics of the objects in proximity
                if relative_distance < 10:

                    # Define plotting parameters
                    name = self.classes[object_class].capitalize()
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Color in RGB format
                    color = (255, 255, 255)

                    info = "{} ({:.0f}m)".format(name, relative_distance)
                    conf = "{0:.0%}".format(mem["confidences"][i])

                    # Adjust x based on the name length
                    x_1 = x + len(info) * 7

                    # Plot figures
                    cv2.rectangle(frame, (x_1, y - 40), (x, y), (0, 0, 0), -1)
                    cv2.putText(frame, info, (x, y - 25), font, 0.4, color, 1)
                    cv2.putText(frame, conf, (x, y - 5), font, 0.4, color, 1)
        return frame

    def predict(self, frame: np.ndarray, mem: dict) -> dict:
        """Predict objects on the frame

        Args:
            frame: Native frame
            mem: Hash table of latest parameters and metrics

        Returns:
            mem: Hash table of updated parameters and metrics
        """
        start_time = time.perf_counter()
        W, H = mem["width"], mem["height"]
        for key in ["indices", "boxes", "confidences", "classes"]:
            mem[key] = []

        # Run prediction on the model
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True)
        self.model.setInput(blob)
        predictions = self.model.forward(self.ln)

        for prediction in predictions:
            for obj in prediction:
                scores = obj[5:]

                # Return the object's class name with the maximum value
                object_class = np.argmax(scores)
                prediction_confidence = scores[object_class]

                # Check if confidence is above the set confidence threshold
                if prediction_confidence > mem["confidence"]:
                    box = obj[0:4] * np.array([W, H, W, H])
                    (x_center, y_center, width, height) = box.astype("int")
                    x = int(x_center - (width / 2))
                    y = int(y_center - (height / 2))

                    # Update parameters in the hash table
                    mem["confidences"].append(float(prediction_confidence))
                    mem["boxes"].append([x, y, int(width), int(height)])
                    mem["classes"].append(object_class)

                    # Perform the non maximum suppression
                    b, c = mem["boxes"], mem["confidences"]
                    confidence, thold = mem["confidence"], mem["threshold"]
                    mem["indices"] = cv2.dnn.NMSBoxes(b, c, confidence, thold)

        # Measure the time it takes to run detection on a single frame
        mem["processing_time"] = (time.perf_counter() - start_time) * 1000
        return mem
