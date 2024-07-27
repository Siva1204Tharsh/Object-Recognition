import numpy as np
import imutils
import cv2
import time 

prototxt="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
confThreshold=0.2

CLASSES=["background","aeroplane","bicycle","bird","boat",
         "bottle","bus","car","cat","chair","cow","diningtable",
         "dog","horse","motorbike","person","pottedplant",
         "sheep","sofa","train","tvmonitor","mobile"]

COLORS=np.random.uniform(0,255,size=(len(CLASSES),3)) #random colors using iamges from internet
print ("Loading network...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print ("model loaded...")
print ("Starting video stream...")

cap = cv2.VideoCapture(0) #Camera  id initialization
time.sleep(2.0)

while(True):
    ret, frame = cap.read()
    frame=imutils.resize(frame,width=1000)
    (h, w) = frame.shape[:2]
    imResizeBlob=cv2.resize(frame,(300,300))
    blob=cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)
    net.setInput(blob)
    detections=net.forward()
    detShape=detections.shape[2]
    for i in range(0,detShape):
        confidence=detections[0,0,i,2]  # 2-confidence of the prediction
        if confidence>confThreshold:
            idx=int(detections[0,0,i,1]) # 1-index of the class
            print("ClassID:",idx)
            print("Confidence:",confidence)

            box=detections[0,0,i,3:7]*np.array([w,h,w,h]) # 3-4-5-6: x1,y1,x2,y2
            (startX,startY,endX,endY)=box.astype("int")

            label="{} {:.2f}".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            if startY -15 >15:
                y =startY-15
            else:
                startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
            
            