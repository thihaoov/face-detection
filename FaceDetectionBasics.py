import cv2
import mediapipe as mp
import time

# capture vd and initialize time
cap = cv2.VideoCapture("FaceVideos/2.mp4")
pTime = 0

# declaring mp classes and objects
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils

faceDetection = mpFaceDetection.FaceDetection(model_selection = 1)

# operation started
while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box # bounding class
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255,0,0), 2) 
            cv2.putText(img, f'FPS: {int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

            

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
    img = cv2.resize(img, (0,0), fx=0.7, fy=0.7)
    cv2.imshow("Face", img)
    cv2.waitKey(30)