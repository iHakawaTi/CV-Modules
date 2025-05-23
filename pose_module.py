import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    """
    Detects and tracks body pose landmarks using MediaPipe's Pose solution.

    Attributes:
        mode (bool): Whether to treat input images as a batch of static images.
        model_complexity (int): Complexity of the pose landmark model (0, 1, or 2).
        enable_segmentation (bool): Whether to enable segmentation mask generation.
        smooth_segmentation (bool): Whether to apply smoothing to the segmentation mask.
        smooth (bool): Whether to smooth the landmark keypoint values over time.
        detectionCon (float): Minimum detection confidence ([0.0, 1.0]).
        trackCon (float): Minimum tracking confidence ([0.0, 1.0]).
    """

    def __init__(self, mode=False, model_complexity=1, enable_segmentation=False,
                 smooth_segmentation=True, smooth=True, detectionCon=0.5, trackCon=0.5):
        """
        Initializes the pose detector with the specified parameters.

        Args:
            mode (bool): Static image mode or video stream mode.
            model_complexity (int): Complexity of the pose model.
            enable_segmentation (bool): Whether to output segmentation masks.
            smooth_segmentation (bool): Whether to smooth segmentation results.
            smooth (bool): Whether to smooth landmark detection over time.
            detectionCon (float): Minimum confidence for initial detection.
            trackCon (float): Minimum confidence for tracking landmarks.
        """
        self.mode = mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity,
                                     self.enable_segmentation, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        """
        Processes the image to detect body landmarks and optionally draw them.

        Args:
            img (ndarray): Input BGR image.
            draw (bool): Whether to draw pose landmarks on the image.

        Returns:
            img (ndarray): Output image with or without landmarks drawn.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Extracts the 2D positions of pose landmarks.

        Args:
            img (ndarray): Image from which pose landmarks are extracted.
            draw (bool): Whether to draw small circles on landmarks.

        Returns:
            lmList (list): A list of landmark positions in the format [id, x, y].
        """
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        """
        Calculates the angle between three pose landmarks.

        Args:
            img (ndarray): Image for drawing.
            p1, p2, p3 (int): Landmark indices.
            draw (bool): Whether to draw the angle visualization.

        Returns:
            angle (float): Angle in degrees between the three points.
        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)

            for point in [(x1, y1), (x2, y2), (x3, y3)]:
                cv2.circle(img, point, 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, point, 15, (0, 0, 255), 2)

            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle


def main():
    """
    Test the pose detector with webcam video stream.
    Draws pose landmarks and prints FPS.
    """
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
