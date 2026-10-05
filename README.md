> Language: English 🇺🇸
![UAV Military Convoy Tracking](https://shieldcn.dev/header/glow.svg?title=UAV+Military+Convoy+Tracking&subtitle=MODEL%3A+YOLOv8s&logo=ri%3APiDrone&mode=dark&font=space-grotesk&image=https%3A%2F%2Fimages.unsplash.com%2Fphoto-1462331940025-496dfbfc7564%3Fw%3D1600%26q%3D70%26fit%3Dcrop%26fm%3Djpg&overlay=1)

For Turkish : [Turkish](README_TR.md)

# Description
![gif1](assets/uav_tracking_demo.gif)<br>

- For this project, i've used the YOLOv8s model to fine-tune. Currently, The fine-tuned model can detect military objects like tanks, military vehicles, trucks and soldiers. Also it can detect explosions which was caused by hit targets.<br>

![gif2](assets/uav_tracking_demo2.gif)

- Another ability of the model is that it can keep the number of the current objects in its memory and display the current situation as a counter:<br>

![cntr](assets/counter-sample.png)

### How does YOLO detect objects?
- YOLO (You Look Only Once) models are computer vision models which were developed by Ultralytics. YOLO models use FCN (Fully Convolutional Network). It simply means there is no Fully Connected Layer in the structure. Instead, 1x1 Convolution filters are used as decision makers.<br>

![conv-filters](assets/1x1-convolution.png)
- Big blue block represents 192 feature maps which were created during the Feature Extraction phase (CNN). The long yellow block indicates 192 sub-filters (these sub-filters are updated in the back-propagation phase), and the last one the green block stands for the final output tensor of the 1x1 Convolution Filter.
- The number of 1x1 Convolution Filters depends on the number of the classes. But first 4 of them always indicates these:  
    - First filter = x deviation
    - Second filter = y deviation
    - Third filter =  width 
    - Fourth filter = height<br>
    
    Remaining filters represent the possibilities of the classes.
- Usually there are three kinds of shapes for output tensors. These are 80x80, 40x40 and 20x20 grids.  
    - 80x80 grids detects small objects
    - 40x40 grids detects mid-sized objects
    - 20x20 grids detects big objects<br>

![grid-sample](assets/grids-sample.png)
> Bigger grids, detect bigger objects
- Once the YOLO model receives an image, it draws 8400 bounding boxes for the image (80x80 Grid = 6400, 40x40 Grid = 1600 , 20x20 Grid = 400) by looking it only once. That is why it is called YOLO (You Only Look Once).
- Then, it chooses the box which has the highest conf value. After that, it eliminates the other boxes according to the IoU (Intersection Over Union) rule. IoU indicates the rate of combination between boxes. (IoU= Intersection / Union)

### Dataset preparation
I've combined two datasets to train the model:<br>
- For the first dataset, i've used a [pipeline](preparing-data/synthetic-data-pipeline.py) to extract frames from an UAV footage. I've taken 2 frames per second (1156 frames in total).
Then, i've used roboflow in order to annotate the objects in each frame. I've applied preprocessing and augmentation to the dataset before download:     
    - Pre-process: **Resize** (640x640,with black edges) 
    - Augmentation: **Flip**(horizontal) , **Crop**(0-15%) , **Exposure**(-9% to 9%)
    - Three outputs were created per training example. Final version of the dataset had 2722 total images (2382 train, 227 validation and 113 test).
- For the second dataset, i've used [Military Footage Dataset](https://universe.roboflow.com/magisterka-gdfg0/military_footage_recognition) to support the available, synthetic dataset. This dataset contains 6149 military convoy images which includes objects from Artillery, Car , Explosion , Military_truck, Military_vecihle, Soldier, Tank  and Truck classes. Preprocess and augmentation steps were applied to this dataset too:  
    - Pre-process: **Resize** (640x640,with black edges)
    - Augmentations: **Flip** (Horizontal), **Crop** (0% minimum, 15% maximum), **Grayscale** (was applied to 35% of images), **Exposure** (-11% to 11%)
    - Two outputs were created per training example. Final version of the dataset had 11191 total images (10084 train, 1002 validation and 105 test)<br>
>[!NOTE]
> As it can be noticed, Grayscale was only applied to the second dataset and only horizontal flip was applied to the both datasets. That's because the first dataset's images already had grayscale images and vertical flip would possibly confuse the model since it turns the images upside-down.
- The second dataset contained eight classes but some of them were unnecessary and were indicating almost the same objects. Since i needed five of them, i've used another [pipeline](preparing-data/dataset-pipeline.py) to cut the Artillery and Car classes. Relevant pipeline deletes images and labels which contains unwanted classes. In addition, *Truck* class was indicating the same object as *Military_truck* class does. Thus, *Truck* class was deleted and its objects were assigned to the *Military_truck* class.

## Training Details

The model was trained for 250 epochs. But patience = 30 parameter stopped it early at 77th epoch. Best results were observed at 47th epoch so it was saved as best.pt. This shows that the model could be way more improved and i could try to diversify both dataset and parameters. However, i think the model's current performance is acceptable even though it is not enough, and could be assumed as a starting point.

**Model**: `YOLOv8s`<br>
**Task**: `Detecting military objects`<br>
**Epoch**: `250 (stopped early at 77)` <br>
**Dataset**: `A hybrid dataset` <br>
**Input Image Size**: `640x640` <br>
**Batch Size**: `8` <br>
**Tech Stack**: ![OpenCV](https://img.shields.io/badge/OpenCV-black?logo=opencv&logoColor=B80B0B
),![Ultralytics](https://img.shields.io/badge/Ultralytics-black?logo=ultralytics&logoColor=0B5FB8
) ![YOLO](https://img.shields.io/badge/YOLO-black?logo=yolo&logoColor=345C11
),![Roboflow](https://img.shields.io/badge/Roboflow-black?logo=roboflow&logoColor=730BB8
)<br>
- There are some vital parameters in terms of YOLO model training:  
    - device="cuda": It is essential to run it on GPU since it is a mid-scaled project. It took three and a half hours to finish on GPU. It would take much longer if i ran it on CPU.<br>
    - batch=8: It helps the GPU to cooldown by reducing the images to be processed at once.<br>
    - patience=30: Stops training process if no improvement seen for a while. It worked in this project.

## Tracking Details  
- Tracking phase required lots of settings. I've applied them one by one to reach the best-looking output of the model.

    - First of all i did not want every found object to have the same box and label color. Thus, i have used a for loop to draw each box manually instead of .plot() function.
    - I have wanted to see how many current object are there in the screen. So i have deployed a counter to each class.
    - Boxes were looking so concrete and distractive that i could not see the objects in the video. Because of that, i have used cv.AddWeight function to add some transparency to the screen. With that, the output video got rid of the messy look.
    - Objects' class names were not stable in certain points of the output video. To solve this, i have used a dictionary structure for each track id to keep their last 20 class ids. I provided model to monitor the most frequent class id for each box. 
    - The output video had no sound since VsCode does not support sounds. To cope with that, i have deployed *moviepy* library in order to copy the sound of the raw video and then transfer it to the output video.

## Model Outputs
- The model was tested in a tracking task. It tried to detect military objects in an UAV Military Convoy Footage (Bayraktar TB2 Arma 3 Simulator). First two outputs can be observed at the very beginning of the project: [Description](#description)
<div align="center">
  <h4>Output 3</h4>
  <img src="assets/uav_tracking_demo4.gif" alt="Description">
</div>
<div align="center">
  <h4>Output 4</h4>
  <img src="assets/uav_tracking_demo3.gif" alt="Description">
</div>
<div align="center">
  <h4>Output 5</h4>
  <img src="assets/uav_tracking_demo1.gif" alt="Description">
</div>

### Installation and usage
- I have fine tuned the model to detect military objects from five classes. So, if you have any video or footage that contains these objects, you can try it by following the steps below.<br>

1-
```bash
#Clone the repo 
git clone https://github.com/halileroglu711/UAV-Military-Convoy-Tracking.git
```
2-
```bash
#Enter the project folder
cd UAV-Military-Convoy-Tracking
```
3-
```bash
#Install requirements
pip install -r requirements.txt
```
4-
```bash
#Start Tracking
python tracking.py
```
### Project Folder Structure
```text
UAV-Military-Convoy-Tracking/

├── assets/
│   ├── 1x1-convolution.png
│   ├── counter-sample.png
│   ├── grids-sample.png
│   ├── uav_tracking_demo.gif
│   ├── uav_tracking_demo1.gif 
│   ├── uav_tracking_demo2.gif
│   ├── uav_tracking_demo3.gif
│   └── uav_tracking_demo4.gif  
│
│
├── preparing-data/
│   ├── dataset-pipeline.py
│   └── synthetic-data-pipeline.py
│
│
├── weights/
│   ├── best.pt
│
├── .gitignore
├── README_TR.md
├── README.md
├── requirements.txt
├── tracking.py
├── training.py
└── UAV-Military-Convoy-Tracking-output.mp4
```

### 💼 License
This project is licensed under the **MIT License**. <br>
Check the [LICENSE](LICENSE) file for further detail.


### 📬 Contact
- Let me know if i have done any mistakes 🙋. Im waiting for your contributions 🙂. Here is where you can find me:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/halil-ero%C4%9Flu-5505783a1)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/halileroglu711)
[![Gmail](https://img.shields.io/badge/Gmail-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:halileroglu711@gmail.com)



