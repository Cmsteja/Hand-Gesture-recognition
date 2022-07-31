from xml.etree.ElementPath import prepare_descendant
import cv2
import time
import os

import HandTrackingModule as htm
import joblib
wCam, hCam = 640, 480

#cap = cv2.VideoCapture(0)
# cap.set(3, wCam)
# cap.set(4, hCam)
model_svm=joblib.load("Svm_joblib")
folderPath = "Hand_Postures"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]
val=[]
i=0
quant=0
prediction=True
how=0
while i<len(overlayList)-1:
    
    img = detector.findHands(overlayList[i],draw=True)
    
    lmList,lmList_x,lmList_y,right,left = detector.findPosition(img, draw=False)
   
    #print(lmList)
    
    ans=[]
    lmList.append(lmList_x)
    lmList.append(lmList_y)
    lists = [lmList_x,lmList_y]
    for j in lmList_y:
        lmList_x.append(j)
    
    ans.append(lmList_x)

       

    if(len(ans[0])):
      how=how+1
      pred_out=model_svm.predict(ans)
      print(pred_out, myList[i])
      print(time.time()-pTime)
      conv=str(myList[i])
      calc=str(pred_out)
      calc=calc[2]
      #print(calc)
      if(calc==conv[0]):
        #prediction=True
        quant=quant+1
    #print(quant)
    
        
    #cv2.waitKey(5)
    i=i+1 

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

   
    cv2.waitKey(1)
print(quant,how)
print((quant/how)*100)