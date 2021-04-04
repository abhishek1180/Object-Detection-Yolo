#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy
import cv2
cap=cv2.VideoCapture(0)

allrows=open('./synset_word.txt').read().strip().split('\n')
classes=[r[r.find(' ')+1:] for r in allrows]
net=cv2.dnn.readNetFromCaffe('./bvlc_googlenet.prototxt','./bvlc_googlenet.caffemodel')

if cap.isOpened()==False:
    print("can't open")
while True:
    ret,frame=cap.read()    # two parameters:return and frame
    blob=cv2.dnn.blobFromImage(frame,1,(224,224))
    net.setInput(blob)
    outp=net.forward()
    r=1
    for i in numpy.argsort(outp[0])[::-1][:5]:
        txt='"%s" probability "%.3f"'%(classes[i],outp[0][i]*100)
        cv2.putText(frame,txt,(0,25+40*r),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        r+=1
    if ret==True:
        cv2.imshow('Frame',frame)
        #  Now, in the case of images, we pass zero to the waitkey function. But for playing a video, we need to pass a number greater than zero to the waitkey function, this is because zero would pause the frame in the video. 
        if cv2.waitKey(25) & 0xFF==27: # 0xFF means escape key which we wait for
            break
    else:
        break
        
# Need to release video capture object and close windows.
cap.release()
cv2.destroyAllWindows()

