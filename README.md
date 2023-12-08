# Cancer-Detection
Lung Cancer Cell Detection


## Introduction
This project aims to develop an automated system for detecting cancerous tissues from histopathological images using advanced machine learning techniques, including convolutional neural networks (CNNs) and YOLOv8 object detection.

### Prerequisites
- Python 3.8 or later
- Access to a CUDA-compatible GPU for training

### Setup
To run this project, first clone the repository and install the required packages using pip:


### Data Preparation 

The dataset used is structured for YOLOv8 object detection. To prepare the data:

   Define the path to the zip file containing the dataset.
   Unzip the dataset to the desired directory.
## Data
The dataset consists of histopathological images annotated with the presence of cancerous cells. Images are preprocessed and resized to a uniform dimension before being fed into the model.

## Model
We use a YOLOv8 model, pre-trained on a general dataset and fine-tuned on our specific histopathological image data.

## Results
Our model achieves:  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 7/7 [00:06<00:00,  1.13it/s]
                     all        211       1081      0.773      0.685      0.761      0.465
                     stas       211       1081      0.773      0.685      0.761      0.465. 
The following are some sample detections:

![Sample Detection 1](Sample_1.png)
![Sample Detection 2](Sample_2.png)

### Training
The model training is performed using the YOLOv8 framework. Training sessions can be customized by varying the number of epochs, batch size, learning rate, and optimizer.!yolo train model=yolov8n.pt data='/content/dataset/data.yaml' epochs=60 verbose=True
Additional trainings with different parameters can be executed as needed:
!yolo train model=yolov8n.pt data='/content/dataset/data.yaml' epochs=120 verbose=True
!yolo train model=yolov8n.pt data='/content/dataset/data.yaml' epochs=80 batch=16 lr0=0.01 optimizer=AdamW verbose=True

### Exporting Results
After training, the model's weights and results can be zipped and downloaded for further analysis.

### License
This project is distributed under the MIT License. See LICENSE for more information.
