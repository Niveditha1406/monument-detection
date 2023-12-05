# Download YOLOv7 repository and install requirements
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="KpbK8p2OBZNYiE59jP4z")
project = rf.workspace("deepstream").project("memorial-w4nlv")
dataset = project.version(1).download("yolov7")

# download COCO starting checkpoint
%cd /content/yolov7
!wget "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"

# run this cell to begin training
%cd /content/yolov7
!python train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 55 --data {dataset.location}/data.yaml --weights 'yolov7.pt' --device 0 

# Run evaluation
!python detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source /content/Statue.mp4

#display inference on ALL test images

import glob
from IPython.display import Image, display

i = 0
limit = 10000 # max images to print
for imageName in glob.glob('/content/yolov7/runs/detect/exp/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1

# optional, zip to download weights and results locally

!zip -r export.zip runs/detect
!zip -r export.zip runs/train/exp/weights/best.pt
!zip export.zip runs/train/exp/*
