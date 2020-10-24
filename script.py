import cv2
import numpy as np 
import math
from centroidtracker import CentroidTracker
from itertools import combinations
import imutils

#Create tracker object
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

#Model files
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"

#Detector load
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

#alarm range 
alarm_dist=75.0

#Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#GREEN
green=(0,255,0)
#RED
red=(0,0,255)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

#Detections
def detect():
    cap=cv2.VideoCapture('video.mp4')
    while True:
        ret,frame=cap.read()
        frame = imutils.resize(frame, width=600)

        #Set the height and width of image
        (H, W) = frame.shape[:2]

        #Create the blob
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        #Perform detections
        detector.setInput(blob)
        detections=detector.forward()

        #Bounding box
        rects=[]

        #Load tracker and track person
        for i in np.arange(0,detections.shape[2]):
            condidence=detections[0,0,i,2]
            if condidence>0.5:
                idx=int(detections[0,0,i,1])

                if CLASSES[idx]!="person":
                    continue

                person_box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        #Configuring bounding boxes
        boundingboxes=np.array(rects)
        boundingboxes=boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        #Centroid dict 
        centroid_dict=dict()
        for box in rects:
            x1,y1,x2,y2=box 
            xmid=(x1+x2)//2
            ymid=(y1+y2)//2
            centroid_dict[(xmid,ymid)]=box

        #alarming list 
        danger=[]
        keys=centroid_dict.keys()
        for x,y in combinations(keys,2):
            x1=x[0]
            y1=x[1]
            x2=y[0]
            x3=y[1]
            dx=x1-x2
            dy=y1-y2
            dist=math.sqrt(dx*dx+dy*dy)
            if dist<alarm_dist:
                danger.append(x)
                danger.append(y)

        for items in keys:
            box=centroid_dict[items]
            if items in set(danger):                
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),red,2)
            else:
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),green,2)

        cv2.imshow("Monitor",frame)
        
        if cv2.waitKey(1)==ord('q'):
            break
    cv2.destroyAllWindows()

#Call detections
detect()