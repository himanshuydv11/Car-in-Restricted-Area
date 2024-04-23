import numpy as np
import pandas as pd
import cv2
import ultralytics
from ultralytics import YOLO
import math
import pygame

model = YOLO('yolov8s.pt')

# Polygon corner points coordinates
area2 = [(400,100), (680,100), (700,480), (400, 480), (400,100)]
area3 = [(400,150), (480,150), (470,180), (400, 180), (400,150)]

pygame.init()
my_sound = pygame.mixer.Sound('siren.wav')

video = cv2.VideoCapture('video3.mp4')
# get FPS of input video 
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
buffer = video.get(cv2.CAP_PROP_BUFFERSIZE)
print(buffer)

classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 
              'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
              'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 
              'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']


while True:
    success, img = video.read()
    results = model(img, stream=True)

    # res = results[0].boxes.data
    # px = pd.DataFrame(res).astype("float")

    # for index, row in px.iterrows():
    #     x1 = int(row[0])
    #     y1 = int(row[1])
    #     x2 = int(row[2])
    #     y2 = int(row[3])
    #     d = int(row[5])
    #     c = classnames[d]

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls = int(box.cls[0])
            cls = classnames[cls]

            if 'car' in cls:
                ans1 = cv2.pointPolygonTest(np.array(area3, np.int32), ((x2,y2)), False)
                ans2 = cv2.pointPolygonTest(np.array(area3, np.int32), ((x1,y2)), False)
                ans3 = cv2.pointPolygonTest(np.array(area3, np.int32), ((x1,y1)), False)
                ans4 = cv2.pointPolygonTest(np.array(area3, np.int32), ((x2,y1)), False)

                if ans1 > 0 or ans2 > 0 or ans3 > 0 or ans4 > 0:
                    # for playing note.wav file
                    my_sound.play()
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255),1,1)
                    cv2.putText(img, f'Car in restricted area zone', (x1,y1), cv2.FONT_HERSHEY_COMPLEX, (0.5),(255,255,255),1)
                else:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),1,1)
                    cv2.putText(img, str(cls), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, (0.5),(255,255,255),1)
    
    cv2.polylines(img, [np.array(area3, np.int32)], True, (0,0,255),2)

    cv2.imshow("Image", img)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()