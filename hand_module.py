import cv2
import mediapipe as mp
import time

class HandDetector():
    """
    Detects and tracks hand landmarks using MediaPipe's Hands module.

    Attributes:
        mode (bool): Static image mode or not.
        max_hands (int): Maximum number of hands to detect.
        model_complexity (int): Complexity of the hand landmark model (0 or 1).
        detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection.
        tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand tracking.
    """

    def __init__(self, mode=False, max_hands=2, model_complexity=1,
                 detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initializes the hand detector with the specified settings.

        Args:
            mode (bool): Whether to treat input images as a batch of static images.
            max_hands (int): Maximum number of hands to detect.
            model_complexity (int): Complexity of the hand landmark model.
            detection_confidence (float): Minimum confidence threshold for detection.
            tracking_confidence (float): Minimum confidence threshold for tracking.
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,
                                        self.model_complexity, self.detection_confidence,
                                        self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        """
        Detects hands and draws landmarks on the given image.

        Args:
            img (ndarray): Input BGR image.
            draw (bool): Whether to draw hand landmarks and connections.

        Returns:
            img (ndarray): Image with or without hand annotations.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """
        Finds the positions of hand landmarks for a specific hand.

        Args:
            img (ndarray): Input BGR image.
            handNo (int): Index of the hand to analyze (0 for first hand).
            draw (bool): Whether to draw circles on each landmark.

        Returns:
            lmList (list): List of landmark positions in the form [id, x, y].
        """
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 25, (0, 0, 255), cv2.FILLED)

        return lmList

# Dummy test code
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        cv2.imshow('Video', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
