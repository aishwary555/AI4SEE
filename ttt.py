import cv2
import cvzone
from ultralytics import YOLO
import numpy as np
import math
import time
from sort import *

# Load video
cap = cv2.VideoCapture("cars.mp4")

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)

# Load YOLO model
model = YOLO("yolov8l.pt")

# Class names for YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load mask
mask = cv2.imread("dss.png")
print(mask.shape)


# Tracking (using sort.py file)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Detection limits
limits = [400, 297, 673, 297]
limits_sec = [60, 400, 673, 400]

# Real world distance between the lines in meters
real_distance = 10  # Update this with the actual distance

totalCounts = []
list_start = {}
list_end = {}

while True:
    success, img = cap.read()
    print(img.shape)

    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass in ["car", "truck", "motorbike", "bus", "bicycle"] and conf > 0.3:
                currentarray = np.array([x1, y1, x2, y2, conf])
                detection = np.vstack((detection, currentarray))

    resultTracker = tracker.update(detection)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.line(img, (limits_sec[0], limits_sec[1]), (limits_sec[2], limits_sec[3]), (0, 255, 0), 5)

    for result in resultTracker:
        x1, y1, x2, y2, ID = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 255, 0))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if ID not in totalCounts:
                totalCounts.append(ID)
                list_start[ID] = time.time()
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        if limits_sec[0] < cx < limits_sec[2] and limits_sec[1] - 15 < cy < limits_sec[3] + 15:
            if ID in totalCounts and ID not in list_end:
                list_end[ID] = time.time()
                cv2.line(img, (limits_sec[0], limits_sec[1]), (limits_sec[2], limits_sec[3]), (0, 255, 0), 5)
                start_time = list_start.get(ID)
                if start_time is not None:
                    elapsed_time = list_end[ID] - start_time
                    speed_mps = real_distance / elapsed_time  # Speed in meters per second
                    speed_kmph = speed_mps * 3.6*10  # Speed in kilometers per hour
                    print(f"Vehicle ID {ID} speed: {speed_kmph:.2f} km/h")
                    cvzone.putTextRect(img, f"ID: {ID}, Speed: {speed_kmph:.2f} km/h", (x1, y1 - 10), scale=1, thickness=2, offset=6)

    cvzone.putTextRect(img, f"Counts: {len(totalCounts)}", (50, 50), scale=1, thickness=2, offset=6)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
        

cap.release()
cv2.destroyAllWindows()
