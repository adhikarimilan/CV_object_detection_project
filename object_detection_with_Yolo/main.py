import numpy as np
from ultralytics import YOLO
import cvzone
from cv2 import cv2
import math
from sort import *

# detection using webcam
# cap=cv2.VideoCapture(0);
# cap.set(3, 640)
# cap.set(4, 480)

# detection of video file
cap = cv2.VideoCapture("./resources/video/cars1.mp4")

model = YOLO('./weights/yolov8s.pt')

classnames = {0: 'person',
              1: 'bicycle',
              2: 'car',
              3: 'motorcycle',
              4: 'airplane',
              5: 'bus',
              6: 'train',
              7: 'truck',
              8: 'boat',
              9: 'traffic light',
              10: 'fire hydrant',
              11: 'stop sign',
              12: 'parking meter',
              13: 'bench',
              14: 'bird',
              15: 'cat',
              16: 'dog',
              17: 'horse',
              18: 'sheep',
              19: 'cow',
              20: 'elephant',
              21: 'bear',
              22: 'zebra',
              23: 'giraffe',
              24: 'backpack',
              25: 'umbrella',
              26: 'handbag',
              27: 'tie',
              28: 'suitcase',
              29: 'frisbee',
              30: 'skis',
              31: 'snowboard',
              32: 'sports ball',
              33: 'kite',
              34: 'baseball bat',
              35: 'baseball glove',
              36: 'skateboard',
              37: 'surfboard',
              38: 'tennis racket',
              39: 'bottle',
              40: 'wine glass',
              41: 'cup',
              42: 'fork',
              43: 'knife',
              44: 'spoon',
              45: 'bowl',
              46: 'banana',
              47: 'apple',
              48: 'sandwich',
              49: 'orange',
              50: 'broccoli',
              51: 'carrot',
              52: 'hot dog',
              53: 'pizza',
              54: 'donut',
              55: 'cake',
              56: 'chair',
              57: 'couch',
              58: 'potted plant',
              59: 'bed',
              60: 'dining table',
              61: 'toilet',
              62: 'tv',
              63: 'laptop',
              64: 'mouse',
              65: 'remote',
              66: 'keyboard',
              67: 'cell phone',
              68: 'microwave',
              69: 'oven',
              70: 'toaster',
              71: 'sink',
              72: 'refrigerator',
              73: 'book',
              74: 'clock',
              75: 'vase',
              76: 'scissors',
              77: 'teddy bear',
              78: 'hair drier',
              79: 'toothbrush'}

mask=cv2.imread("./resources/image/mask.png")

#tracking using sort library
tracker=Sort(max_age=20,min_hits=3, iou_threshold=0.3)

#boundry line co-ordinates from where the counting begins
line_x_y=[450,450,1600,450]
total_count=[]
while True:
    success, img = cap.read()
    detection_region=cv2.bitwise_and(img,mask)
    results = model(detection_region, stream=True)

    detections=np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)

            # confidence
            conf = math.ceil((box.conf[0] * 100))
            #print(conf)

            # classname
            cls = box.cls[0]
            cls = int(cls)

            #only detecting road vehicles
            if cls in {1,2,3,5,6,7} :
            # cvzone.putTextRect(img, f'{classnames[cls]} {conf}', (max(0, x1), max(25, y1-20)), colorT=(255,255,0), colorB=(0,0,0), thickness=1, offset=2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
                cv2.putText(img, f'{classnames[cls]} {conf}', (max(0, x1), max(10, y1 - 10)), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255))

                current_array=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,current_array))


            # alpha = 0.4  # Transparency factor.

            # Following line overlays transparent rectangle
            # over the image
            # image_new = cv2.addWeighted(img, alpha, img, 1 - alpha, 0)

    #Now time to track each vehicle
    results_tracker=tracker.update(detections)

    #drawing a red line boundry to count the vehicle
    cv2.line(img,(line_x_y[0],line_x_y[1]),(line_x_y[2],line_x_y[3]),(0,0,255),4)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1,y1,x2,y2, id=int(x1),int(y1),int(x2),int(y2), int(id)
        w,h=x2-x1,y2-y1
        cx,cy=x1+w//2,y1+h//2
        cx,cy=int(cx), int(cy)
        #cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        #if only the line's coordinates and the vehicle's centre point intersect
        if line_x_y[0]<cx<line_x_y[2] and line_x_y[1]-15<cy<line_x_y[3]+15:
            if total_count.count(id)==0:
                total_count.append(id)
            cvzone.putTextRect(img, f'+ 1', (cx, cy))
            cv2.line(img, (line_x_y[0], line_x_y[1]), (line_x_y[2], line_x_y[3]), (0, 255, 0), 4)

        cvzone.putTextRect(img,f'Count: {len(total_count)}',(50,50))
    cv2.imshow("Vehicle Counter", img)
    #cv2.imshow("Detection region", detection_region)
    #cv2.waitKey(0)
    # Press Esc to exit the loop
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
