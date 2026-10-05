#In order to diversify the ready-to-use dataset,i am going to extract some frames from a UAV Convoy Tracking video
import cv2 as cv
import os
path="preparing-data/original-video.mp4"
cpt=cv.VideoCapture(path)
frame_cnt=0

saved_cnt=0
while cpt.isOpened():
    succes,frame=cpt.read()
    if not succes:
        break
    if frame_cnt % 30 == 0:
        name=os.path.join("preparing-data/gathered-frames",f"synt_frame_{saved_cnt}.jpg")
        cv.imwrite(name,frame)
        saved_cnt+=1
    frame_cnt+=1
cpt.release()
print(f"The task has been finished. Synthetic datas have been produced.")
    
