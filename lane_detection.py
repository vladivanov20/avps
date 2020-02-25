"""
Author: Vladyslav Ivanov
Contact Email: vladyslav.iva@gmail.com
File Description: Lane Detection with Hough transform
"""
import numpy as np
import collections
import cv2


class Road:
    def __init__(self):
        # Create hash table to keep track of data
        self.mem = collections.defaultdict(int)

        # Line parameters
        for key in ["left_m", "right_m", "left_b", "right_b", "opening"]:
            self.mem[key] = []

    def roi(self, frame):
        """Crop frame to the region of interest

        Args:
            frame: Native frame

        Returns:
            Cropped frame with pixels that matched the mask
        """
        w = int(frame.shape[1])
        h = int(frame.shape[0])

        # Define region of interest's vertices
        vertices = np.array(
            [
                [int(0.10 * w), int(h)],
                [int(0.92 * w), int(h)],
                [int(0.53 * w), int(0.65 * h)],
                [int(0.40 * w), int(0.65 * h)],
            ]
        )

        # Determine color of the mask
        if len(frame.shape) > 2:
            num_channels = frame.shape[2]
            mask_color = (255,) * num_channels
        else:
            mask_color = 255

        # Create blank matrix with frame's dimensions
        mask = np.zeros_like(frame)

        # Fill the polygon (region of interest)
        cv2.fillPoly(mask, np.int32([vertices]), mask_color)
        return cv2.bitwise_and(frame, mask)

    def preprocess(self, frame):
        """ Apply masks, grayscale, and smoothing and run canny edge detection

        Args:
            frame: Current frame

        Returns:
            Edge frame
        """
        # Define color ranges for yellow and white lanes
        white_low = np.array([0, 190, 0])
        yellow_low = np.array([10, 0, 90])
        white_high = np.array([255, 255, 255])
        yellow_high = np.array([50, 255, 255])

        # Convert to HLS color space
        hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

        # Create color-relevant masks
        yellow_mask = cv2.inRange(hls, yellow_low, yellow_high)
        white_mask = cv2.inRange(hls, white_low, white_high)

        # Calculate combined mask and masked image
        mask = cv2.bitwise_or(yellow_mask, white_mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian smoothing with a kernel size of 5
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return cv2.Canny(frame, 50, 120)

    def draw_lane(self, frame, mask, lines):
        """Compute coordinates of the lines and generate the lane overlay

        Args:
            mask: Blank matrix with edge frame's dimensions
            frame: Native frame
            lines: Line segments from Hough transform

        Returns:
            overlay: Lane overlay
        """
        def update(x2, y2, m, direction):
            """Update line segment equation parameters

            Args:
                x2: Second x-coordinate of the line
                y2: Second y-coordinate of the line
                m: Slope of the line equation
                direction: Direction of the line segment

            Returns:
                None
            """
            b = y2 - (m * x2)
            b_data = self.mem[str(direction) + "_b"]
            m_data = self.mem[str(direction) + "_m"]
            b_data.append(b)
            m_data.append(m)

        # Create copy of the native frame to use as an overlay
        overlay = frame.copy()
        try:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x1 - x2 != 0:
                        m = (y1 - y2) / (x1 - x2)
                        # Check direction of the line
                        if m > 0.3:
                            if x1 > 500:
                                update(x2, y2, m, "right")
                            else:
                                None
                        elif m < -0.3:
                            if x1 < 600:
                                update(x2, y2, m, "left")

            # Calculate average slope, intercept, and coordinates of the lines
            left_m_avg = np.mean(self.mem["left_m"][-60:])
            left_b_avg = np.mean(self.mem["left_b"][-60:])
            right_m_avg = np.mean(self.mem["right_m"][-60:])
            right_b_avg = np.mean(self.mem["right_b"][-60:])
            left_x1 = int((0.67 * mask.shape[0] - left_b_avg) / left_m_avg)
            left_x2 = int((mask.shape[0] - left_b_avg) / left_m_avg)
            right_x1 = int((0.67 * mask.shape[0] - right_b_avg) / right_m_avg)
            right_x2 = int((mask.shape[0] - right_b_avg) / right_m_avg)

            # Calculate the width of the lane at its end
            width = right_x1 - left_x1

            # Get average width of the lane end
            width_avg = np.mean(self.mem["w"][-60:]) if self.mem["w"] else None

            # Check for potential outliers
            if not width_avg or width < 2 * width_avg:

                cv2.line(
                    frame,
                    (left_x1, int(0.67 * mask.shape[0])),
                    (left_x2, int(mask.shape[0])),
                    (0, 255, 0),
                    3,
                )

                cv2.line(
                    frame,
                    (right_x1, int(0.67 * mask.shape[0])),
                    (right_x2, int(mask.shape[0])),
                    (0, 255, 0),
                    3,
                )

                vertices = np.array(
                    [
                        [left_x1, int(0.67 * mask.shape[0])],
                        [left_x2, int(mask.shape[0])],
                        [right_x2, int(mask.shape[0])],
                        [right_x1, int(0.67 * mask.shape[0])],
                    ], np.int32,
                    )

                cv2.fillPoly(overlay, [vertices], (0, 255, 255))
                self.mem["w"].append(width)
        except Exception:
            pass
        return overlay

    def get_lane(self, frame, edge, rho, theta, threshold, minLength, maxGap):
        """Create lane from Hough transformation on canny edge detector

        Args:
            frame: Native frame
            edge: Edge frame (8-bit, single-channel binary source image)
            rho: Distance resolution of the accumulator in pixels
            theta: Angle resolution of the accumulator in radians
            threshold: Only lines that get enough votes (> THOLD) are returned
            minLength: Line segments shorter than this length are rejected
            maxGap: Maximum gap between points on the same line to link them

        Returns:
            Overlay with the drawn lane
        """
        # Perform Hough transform
        lines = cv2.HoughLinesP(
            edge,
            rho,
            theta,
            threshold,
            np.array([]),
            minLineLength=minLength,
            maxLineGap=maxGap,
        )

        # Create blank matrix with edge frame's dimensions
        dim = (edge.shape[0], edge.shape[1], 3)
        mask = np.zeros(dim, dtype=np.uint8)
        return self.draw_lane(frame, mask, lines)

    def detect(self, frame):
        """Display detected objects

        Args:
            frame: Native frame
            mem: Hash table of latest parameters and metrics

        Returns:
            mem: Hash table of updated parameters and metrics
        """
        region_of_interest = self.roi(frame)
        edge = self.preprocess(region_of_interest)
        lane = self.get_lane(frame, edge, 1, np.pi / 180, 10, 20, 5)

        # Overlay lane with the original frame
        frame = cv2.addWeighted(lane, 0.7, frame, 0.3, 0)
        return frame
