import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    """
    Detects facial landmarks using MediaPipe's Face Mesh.

    Args:
        staticMode (bool): If True, treats input images as static.
        maxFaces (int): Max number of faces to detect.
        refine_landmarks (bool): Whether to refine landmark locations.
        minDetectionCon (float): Minimum confidence for initial detection.
        minTrackCon (float): Minimum confidence for tracking landmarks.
    """

    def __init__(self, staticMode=False, maxFaces=2, refine_landmarks=False, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refine_landmarks = refine_landmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.refine_landmarks, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds face meshes in the input image.

        Args:
            img (ndarray): Input BGR image.
            draw (bool): Whether to draw mesh on image.

        Returns:
            img (ndarray): Image with or without drawings.
            faces (list): List of faces with landmark coordinates.
        """
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                    face = []
                    for id,lm in enumerate(faceLms.landmark):
                        ih, iw, ic = img.shape
                        x,y = int(lm.x*iw), int(lm.y*ih)
                        face.append([x,y])
                        faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        #if len(faces)!= 0:
            #print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
