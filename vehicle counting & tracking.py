import cv2
from ultralytics import YOLO
from collections import defaultdict


model = YOLO('yolov11l.pt')

class_list =  model.names

cap = cv2.VideoCapture('Teste/4.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    cv2.imshow("YOLO Object Tracking & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()