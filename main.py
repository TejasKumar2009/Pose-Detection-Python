import cv2 as cv
import mediapipe as mp
import time


capture = cv.VideoCapture(0)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

start_time = time.time()
frame_counter = 0

while True:
    isTrue, frame = capture.read()
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = pose.process(frameRGB)

    if (results.pose_landmarks):
            mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    
    frame_counter += 1
    fps = frame_counter/(time.time()-start_time)
    fps = int(fps)

    cv.putText(frame, f"{fps} FPS", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv.imshow("Pose Detection Model", frame)
    
    key = cv.waitKey(1)
    if key == 27:
        break

capture.release()
capture.destroyAllWindows()
