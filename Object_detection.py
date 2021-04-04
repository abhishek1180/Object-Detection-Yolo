#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np



# set conf, nms thresholds,input width and height
Confidence_threshold = 0.25
Non_maximum_suppression_threshold = 0.25
inpWidth = 416
inpHeight = 416


#Load names of classes and turn that into a list
file_classes= "./coco.names"
classes = None

with open(file_classes,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

#Set up the net
net = cv2.dnn.readNetFromDarknet('./yolov3.cfg', './yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIDs = []
    confidences = []
    boxes = []


    

    for out in outs:
        for detection in out:
            
            scores = detection [5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > Confidence_threshold:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)

                width = int(detection[2]* frameWidth)
                height = int(detection[3]*frameHeight )

                left = int(centerX - width/2)
                top = int(centerY - height/2)

                classIDs.append(classID)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with to lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, Confidence_threshold , Non_maximum_suppression_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        drawPred(classIDs[i], confidences[i], left, top, left + width, top + height)

#Process inputs
winName = 'DL OD with OpenCV'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
cv2.resizeWindow(winName, 1000,1000)

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output_videos.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, 
                             (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
if cap.isOpened()==False:
    print("can't open")
while True:
    ret,frame=cap.read()    # two parameters:return and frame
    # Create a 4D blob from a frame.
    blob=cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop = False)
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs=net.forward(getOutputsNames(net))
    # Remove the bounding boxes with low confidence
    postprocess (frame, outs)
    out.write(frame.astype(np.uint8))
    if ret==True:
        cv2.imshow(winName, frame)
        if cv2.waitKey(25) & 0xFF==27: # 0xFF means escape key which we wait for
            break
    else:
        break
        
# Need to release video capture object and close windows.
cap.release()
out.release()
cv2.destroyAllWindows()

