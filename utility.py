"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: System's user interface
"""
import cv2


class UI:
    def __init__(self):
        # Colors
        self.color_title = (102, 102, 100)
        self.color_value = (255, 255, 255)
        self.color_exit = (247, 160, 60)
        self.color_underlay = (0, 0, 0)

        # Fonts
        self.font_title = cv2.FONT_HERSHEY_DUPLEX
        self.font_value = cv2.FONT_HERSHEY_SIMPLEX

        # Overlay parameters
        self.alpha = 0.8

    def display(self, frame, mem: dict):
        """ Display elements of the user interface
        Args:
            frame: Current frame
            type frame: numpy.ndarray
            mem: Hash table of latest parameters and metrics

        Returns:
            frame: Frame with a user interface overlay
            rtype frame: numpy.ndarray
        """
        # Create mask to overlay the interface
        mask = frame.copy()

        # Exit button and metrics underlays
        cv2.rectangle(
            mask,
            (frame.shape[1] - 230, 33),
            (frame.shape[1] - 78, 57),
            self.color_exit,
            -1,
        )

        cv2.rectangle(
            mask,
            (0, 0),
            (int(frame.shape[1] / 2.3), int(frame.shape[0] / 9.5)),
            self.color_underlay,
            -1,
        )

        # Overlay dashboard elements
        frame = cv2.addWeighted(mask, self.alpha, frame, 1 - self.alpha, 0)

        # Exit button
        cv2.putText(
            frame,
            "Press ESC to exit",
            (frame.shape[1] - 225, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.color_value,
            1,
            cv2.LINE_AA,
        )

        # FPS counter
        cv2.putText(
            frame,
            "FPS",
            (27, 30),
            self.font_title,
            0.4,
            self.color_title,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            "{}".format(mem["fps"]),
            (27, 50),
            self.font_value,
            0.4,
            self.color_value,
            1,
            cv2.LINE_AA,
        )

        # Inference time
        cv2.putText(
            frame,
            "Inference time",
            (72, 30),
            self.font_title,
            0.4,
            self.color_title,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "{0:.2f} ms/frame".format(mem["processing_time"]),
            (72, 50),
            self.font_value,
            0.4,
            self.color_value,
            1,
            cv2.LINE_AA,
        )

        # Frames queued counter
        cv2.putText(
            frame,
            "Frames queued",
            (207, 30),
            self.font_title,
            0.4,
            self.color_title,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "{}".format(mem["q-size"]),
            (207, 50),
            self.font_value,
            0.4,
            self.color_value,
            1,
            cv2.LINE_AA,
        )

        cv2.line(
            frame,
            (330, 15),
            (330, int(frame.shape[0] / 9.5) - 15),
            self.color_title,
            1
        )

        # Current video mode
        cv2.putText(
            frame,
            "Mode",
            (357, 30),
            self.font_title,
            0.4,
            self.color_title,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "{}".format(mem["display_mode"]),
            (357, 50),
            self.font_value,
            0.4,
            self.color_value,
            1,
            cv2.LINE_AA,
        )

        # Video playback speed
        cv2.putText(
            frame,
            "Playback speed",
            (427, 30),
            self.font_title,
            0.4,
            self.color_title,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "{}%".format(110 - mem["time"]),
            (427, 50),
            self.font_value,
            0.4,
            self.color_value,
            1,
            cv2.LINE_AA,
        )
        return frame
