import cv2
import cvzone
from ultralytics import YOLO
import math
import time

from sort import *

cap = cv2.VideoCapture("cars.mp4")


#cap.set(3,1280)            #for video we can not set the size
#cap.set(4,720)     


cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Allows window resizing by the user
cv2.resizeWindow("Image", 1280, 720)  # Set to your desired or the video's dimensions



model = YOLO("yolov8l.pt")


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

mask = cv2.imread("dss.png")


list_start = []
list_end = []
#Tracking       (using sort.py file) 
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limits = [400,297,673,297]
limits_sec = [60,400,673,400]
totalCounts = []

while True:
    success,img  = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion , stream =True)
    
    detection = np.empty((0,5))
    
    print(img.shape)
    print(mask.shape)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            
            #BOunding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2) 
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            w,h = x2-x1 , y2-y1
            
            #confidence Match
            conf =math.ceil((box.conf[0]*100))/100     #because we want confidence upto 2 decimal places
            print(conf)
            #Class Name
            cls = int(box.cls[0])
            currentclass = classNames[cls]
            print(currentclass)
            
            if(currentclass == "car" or currentclass == "truck" or currentclass == "motorbike" or currentclass == "bus" or currentclass =="bicycle" and conf > 0.3):
                
#(not always)   cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=0.6,thickness=1,offset=3)
#               cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5,colorR=(0,255,255))           #here l is lenght og bounding box and rt is rectangle thickness
                
                currentarray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((currentarray,detection))
                
    resultTracker = tracker.update(detection)  
    
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    cv2.line(img,(limits_sec[0],limits_sec[1]),(limits_sec[2],limits_sec[3]),(0,255,0),5)
    


    for result in resultTracker:
        x1,y1,x2,y2,ID = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(result)
        
        w,h = x2-x1 , y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=2,colorR=(255,255,0))           #here l is lenght of bounding box and rt is rectangle thickness
        cvzone.putTextRect(img,f' {int(ID)}, {currentclass}',(max(0,x1),max(35,y1)),scale=1,thickness=2,offset=6)
                
                
        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)    #here 5 is radius and cv2.FILLED is thickness
        
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[3] + 15:
            if totalCounts.count(ID) == 0:                 # if id is not there in totalcounts then we will append it to totalCounts
                totalCounts.append(ID)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
        
        
            dx,dy = x1+w//2,y1+h//2
            if limits_sec[0] < dx < limits_sec[2] and limits_sec[1] - 15 < dy < limits_sec[3] + 15:
                cv2.line(img,(limits_sec[0],limits_sec[1]),(limits_sec[2],limits_sec[3]),(0,255,0),5)
            
  
        cvzone.putTextRect(img,f"counts :{len(totalCounts)}",(50,50))
                
    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion",imgRegion)
    cv2.waitKey(0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # here Yellow line shows yolo model detection
    # here light blue line shows object tracking (using sort.py file)