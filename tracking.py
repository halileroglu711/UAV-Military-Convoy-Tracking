from ultralytics import YOLO
import cv2 as cv
import os
from moviepy import VideoFileClip
from collections import deque,Counter
trained_model=YOLO(r"weights/best.pt")

#video capture
path=r"raw/raw_first_01.mp4"
cpt=cv.VideoCapture(path)

#Get the video's features
width=int(cpt.get(cv.CAP_PROP_FRAME_WIDTH))
height=int(cpt.get(cv.CAP_PROP_FRAME_HEIGHT))
fps=cpt.get(cv.CAP_PROP_FPS)
out=cv.VideoWriter("Temp-UAV-Convoy-Tracking.avi",cv.VideoWriter_fourcc(*"XVID"),fps,(width,height))

#Algorithm to ignore temporary classes
previous_classes={}
max_frames=15

#Determine output window
cv.namedWindow("UAV-Military Convoy Tracking",cv.WINDOW_NORMAL)
cv.resizeWindow("UAV-Military Convoy Tracking",1280,720)
header="                IN SIGHT\n|--------------------------|"
#Tracking
while cpt.isOpened():
    explosion=0
    military_truck=0
    military_vehicle=0
    soldier=0
    tank=0
    success,frame=cpt.read()
    if not success:
        break
    results=trained_model.track(
        frame,
        persist=True,
        conf=0.5,
        iou=0.5,
        tracker="bytetrack.yaml",
        verbose=False
    )
    annotated_frame=results[0]
    h,w=frame.shape[:2]
    copy=frame.copy()#get a copy of frame
    alpha=0.6 #transparent rate 
    if annotated_frame.boxes is not None:
        for box in annotated_frame.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            if box.id is not None:
                track_id=int(box.id[0])
                raw_cls_id=int(box.cls[0])
                if track_id not in previous_classes:
                    previous_classes[track_id]=deque(maxlen=max_frames)
                previous_classes[track_id].append(raw_cls_id)
                
                #Find the most repeated class id for the box
                frequent_id=Counter(previous_classes[track_id]).most_common(1)[0][0]
                if frequent_id == 0:
                                cv.rectangle(copy,(x1,y1),(x2,y2),(153,153,255),2)
                                cv.putText(copy,trained_model.names[0],(x1,max(0,y1-10)),cv.FONT_HERSHEY_DUPLEX,0.5,(153,153,255),2)
                                explosion+=1
                elif frequent_id == 1:
                                cv.rectangle(copy,(x1,y1),(x2,y2),(0,204,204),2)
                                cv.putText(copy,trained_model.names[1],(x1,max(0,y1-10)),cv.FONT_HERSHEY_DUPLEX,0.5,(0,204,204),2)
                                military_truck+=1
                elif frequent_id == 2:
                                cv.rectangle(copy,(x1,y1),(x2,y2),(0,204,204),2)
                                cv.putText(copy,trained_model.names[2],(x1,max(0,y1-10)),cv.FONT_HERSHEY_DUPLEX,0.5,(0,204,204),2)
                                military_vehicle+=1
                elif frequent_id == 3:
                                cv.rectangle(copy,(x1,y1),(x2,y2),(0,0,255),2)
                                cv.putText(copy,"soldier",(x1,max(0,y1-10)),cv.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)
                                soldier+=1
                else:
                                cv.rectangle(copy,(x1,y1),(x2,y2),(0,204,204),2)
                                cv.putText(copy,trained_model.names[4],(x1,max(0,y1-10)),cv.FONT_HERSHEY_DUPLEX,0.5,(0,204,204),2)
                                tank+=1
    #Blend boxes
    cv.addWeighted(copy,alpha,frame,1-alpha,0,frame)
    cv.putText(
            frame,
            f"{header}\nHit Targets:                       {explosion}\nMilitary Trucks:               {military_truck}\nMilitary Vehicles:            {military_vehicle}\nSoldiers:                             {soldier}\nTanks:                                  {tank}",
            (1550,h-1000),
            cv.FONT_HERSHEY_DUPLEX,
            0.9,
            (102,102,255),
            2,
            cv.LINE_AA
        )            
    #Monitoring
    cv.imshow("UAV-Military Convoy Tracking",frame)
    out.write(frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cpt.release()
out.release()
cv.destroyAllWindows()

#Output video contains no sound because of vs code's video capturing structrue. Let's add audio to the output video.
try:
    raw_clip=VideoFileClip(r"raw/raw_first_01.mp4")
    processed_clip=VideoFileClip("Temp-UAV-Convoy-Tracking.avi")
    final_video=processed_clip.with_audio(raw_clip.audio)
    final_video.write_videofile("UAV-Convoy-Tracking.avi", codec="libx264", audio_codec="aac")
    
    raw_clip.close()
    processed_clip.close()
    
    print(f"Done. Original audio was added to the output video.")
except Exception as e:
    print(f"The task has failed: {e}")
    

    
                
                
                
                
                
             
    
    


