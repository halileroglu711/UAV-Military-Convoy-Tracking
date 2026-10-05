from ultralytics import YOLO
import cv2 as cv
model= YOLO("yolov8s.pt")
def train():
    model.train(
        data="data/data.yaml", #use datas from data.yaml
        epochs=250,
        imgsz=640, #input image size
        batch=8,
        patience=30,
        name="UAV-Military-Convoy-Tracking-model",#final folder name
        device="cuda", #run it on GPU
        save=True, #save the model's weights
        save_period=50,#save the model's weights every 50 epochs
        val=True,#apply validation after each epoch
        verbose=True,#monitor training process in terminal
        hsv_s=0.0,#saturation manipulation.
        cache=True,
        optimizer='AdamW',
        lr0=0.001,#starting learning rate
        lrf=0.01, #final learning rate
        cos_lr=True,#Apply Coslr for a slower lr decrease
        hsv_v=0.3,#brightness manipulation.
        degrees=0.0,#simulation for the UAV's camera rotations
        translate=0.2,#location manipulation.
        scale=0.1,#zoom-in and zoom-out
        mosaic=1.0,#strenghten the model to detect small objects 
        erasing=0.0#the model can detect the object only seeing a small part of it.
    )
if __name__=="__main__":
    train()