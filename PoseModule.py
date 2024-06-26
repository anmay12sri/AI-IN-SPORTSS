# import cv2
# import mediapipe as mp
# import time
# import math


# class poseDetector():

#     def __init__(self, mode=False, upBody=False, smooth=True,
#                  detectionCon=0.5, trackCon=0.5):

#         self.mode = mode
#         self.upBody = upBody
#         self.smooth = smooth
#         self.detectionCon = detectionCon
#         self.trackCon = trackCon

#         self.mpDraw = mp.solutions.drawing_utils
#         self.mpPose = mp.solutions.pose
#         self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

#     def findPose(self, img, draw=True):
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         self.results = self.pose.process(imgRGB)
#         if self.results.pose_landmarks:
#             if draw:
#                 self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
#                                            self.mpPose.POSE_CONNECTIONS)
#         return img

#     def findPosition(self, img, draw=True):
#         self.lmList = []
#         if self.results.pose_landmarks:
#             for id, lm in enumerate(self.results.pose_landmarks.landmark):
#                 h, w, c = img.shape
#                 # print(id, lm)
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 self.lmList.append([id, cx, cy])
#                 if draw:
#                     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
#         return self.lmList

#     def findAngle(self, img, p1, p2, p3, draw=True):

#         # Get the landmarks
#         x1, y1 = self.lmList[p1][1:]
#         x2, y2 = self.lmList[p2][1:]
#         x3, y3 = self.lmList[p3][1:]

#         # Calculate the Angle
#         angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
#                              math.atan2(y1 - y2, x1 - x2))
#         if angle < 0:
#             angle += 360

#         # print(angle)

#         # Draw
#         if draw:
#             cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
#             cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
#             cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
#             cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
#             cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
#             cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
#         return angle

# def main():
#     cap = cv2.VideoCapture('/Users/prakashvishal93/Downloads/L-Online/AI _Projects/AI-PROJECT-AI-IN-SPORTS-/4.mp4')
#     pTime = 0
#     detector = poseDetector()
#     while True:
#         success, img = cap.read()
#         img = detector.findPose(img)
#         lmList = detector.findPosition(img, draw=False)

#         if len(lmList) != 0:
#             print(lmList[12])
#             cv2.circle(img, (lmList[12][1], lmList[12][2]), 10, (255, 0, 0), cv2.FILLED)

#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime

#         cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
#                     (255, 0, 0), 3)

#         cv2.imshow("Image", img)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
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
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

    def findSpeed(self, lmList, prev_lmList, fps):
        speeds = []
        if len(lmList) == len(prev_lmList) and fps > 0:
            for i in range(len(lmList)):
                id, cx, cy = lmList[i]
                _, prev_cx, prev_cy = prev_lmList[i]
                distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                speed = distance * fps
                speeds.append(speed)
        return speeds

    def findAccuracy(self, lmList, ref_lmList):
        if len(lmList) != len(ref_lmList):
            return 0
        total_error = 0
        for i in range(len(lmList)):
            id, cx, cy = lmList[i]
            _, ref_cx, ref_cy = ref_lmList[i]
            error = math.sqrt((cx - ref_cx)**2 + (cy - ref_cy)**2)
            total_error += error
        accuracy = 1 - (total_error / (len(lmList) * 100))
        accuracy = max(0, accuracy)
        return accuracy

def main():
    cap = cv2.VideoCapture('/path/to/video.mp4')
    pTime = 0
    detector = poseDetector()
    prev_lmList = []
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            if len(prev_lmList) != 0:
                speeds = detector.findSpeed(lmList, prev_lmList, 1 / (time.time() - pTime))
                print("Speeds:", speeds)
            prev_lmList = lmList
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

