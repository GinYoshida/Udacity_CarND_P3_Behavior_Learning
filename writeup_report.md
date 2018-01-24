# **Behavioral Cloning** 
## Writeup report
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./fig/Hist_with_CSV.png  "Hist before augmentation"
[image2]: ./fig/Hist_after_aug.png "Hist after augmentation"
[image3]: ./fig/Example_after_aug.png "Recovery Image"
[image4]: ./fig/Train.png  "RMS data during training"
[image5]: ./fig/train_Example_left.jpg "training data example of left camera"
[image6]: ./fig/train_Example_center.jpg "training data example of center camera"
[image7]: ./fig/train_Example_right.jpg "training data example of right camera"
[image8]: ./fig/Example_dirt.jpg "example of dirt on right"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted 

#### 1. Submission of codes and report  

My project includes the following files:  
* "model.py" containing python code 
* "Code_test.ipynb": jupyter notebook to test each code to create model.py and visualize output
* drive.py: for driving the car in autonomous mode
* model.h5: trained model file: output from model.py 
* writeup_report.md: summarizing the results (This file)

#### 2. Submission of binary files  
With following command, acutonomous driving is possible with simulator.  
```sh
python drive.py model.h5
```  
As evidence, recorded video is included in this repository as "run1.mp4".

---
### Model Architecture and Training Strategy  


#### 1. Basic strategy  
 Based on training material in this project of Udacity and following blog, transfer learning with lower level feature for this project was applied.  
 Because, "small data set" and "Different data set" were seems to be this takes' type.  

 This is Link of blog, which shows transfer learning types.  
 https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2  

 GoogleNet, InceptionV3 in Keras "Applications", was applied with only lower level feature, i.e. until Mixed1 layter.  
 
 This is link for Keras document.  
 https://keras.io/ja/applications/#inceptionv3  

 Model architecture is shown as below.  
 
**Table. Model architecture**  

| Number        | Layer type           | Output size  |
|:-------------:|:-------------:| :-----:|
| 1      | Normalize (Lambda)  | (160, 320, 3) |
| 2      | Cropping (Cropping2D)  | (90, 320, 3) |
| .      | .  |  |
| .      | .  |  |
| .      | .  |  |
| .      | InceptionV3 from 1st to 65th layer  | (8, 37, 288) |
| .      | .  |  |
| .      | .  |  |
| .      | .  |  |
| 3     | Convolution (Conv2D) | (6, 35, 20) |
| 4     | (BatchNormalization) | (6, 35, 20) |
| 5     | (MaxPooling2D) | (3, 17, 20) |
| 6     | Activation(ELU) | (3, 17, 20) |
| 7     | (Dropout)  | (3, 17, 20) |
| 8     | (Flatten)   | (1020) | 
| 9     | Fully-connected (Dense)  | (200) | 
| 10    | (BatchNormalization)  | (200) | 
| 11    | Activation(ELU)  | (200) | 
| 12    | (Dropout)  | (200) | 
| 13    | Fully-connected (Dense)  | (100) | 
| 14    | (BatchNormalization)  | (100) | 
| 15    | Activation(ELU)  | (100) | 
| 16    | (Dropout)  | (100) | 
| 17    | Fully-connected (Dense)  | (1)) | 

Total params: 995,449  
Trainable params: 287,201  
Non-trainable params: 708,248  

**High parameters**  
Number of epochs: 50  
Initial learning rate: 0.005  
batch size: 32  
Keep probability of dropof: 0.6  


**Fig. Training history**  
 Before implementation of batch normalization, "all steering angles = 0" were learned by model.   
 With implmentation of batch normalization, the traing model is stable.  
 RMS of training and validation was decreased straightforward.  

![alt text][image4]   
   
 
 ---
#### 2.Solution for each key point  
 Following points are difficulty to reach the goal.  
 
#### 1) Data collection  

 The driving record was created based on keyboard input only, not with mouse operation. With this operation, almost input of steering angle is 25 or 0. To adjust from very discrete data to continuous data, I added final column with the following operation.  

 If |(Moving average of steering angle with 5 rows)| > |steering angle|, apply moving average. Another case is opposite way.  

 Also, reduction of 0 angle data was implemented during import training data.
 With changing how much 0 angle delete, this characterize the vehicle behaviro. For instance, no reduction of 0 angle data, the vehicle tends to go straight in corner but keep more center part of road in straight part.
 (In model.py: Line 35~37)
 
 Histogram of original steering data is shown as below.  

**Fig. Histogram of training data without data augmentation**  

![alt text][image1]  

#### 2) Data argumentation for angle  

With only center camera data, almost steering angle is 0. It is difficult for model to learn how to control steering.  
  Based on NVIDIA document, 2 data augmentation methods were applied.
  
  a) Apply Left / Right camera data with steering angle of +/- 0.2.  
  (In model.py: Line 46~59)

**Fig. traing data example from each camera**

(a) Left  
![alt text][image5]

(b) Center  
![alt text][image6]

(c) Right  
![alt text][image7]

  b) Apply random shift, from -25 to +25 in horizontal direction, to original training data.
  Then, add (0.2) x (Shifted pixel) / 25 to original steering angle. 
  (In model.py: Line 79~114)

Ref: "2 Overview of the DAVE-2 System" in the following document.
https://arxiv.org/pdf/1604.07316.pdf

 Histogram of training data after data augmentation is shown as below.

**Fig. Histogram of training data with data augmentation**
![alt text][image2]

#### 3) Memory management  

 With augmented data, it is not possible bsecause whole traing size is too big to handl normal fit method of Keras.  
 In accordance with training material of this project, genetator was implemented for the model.  
(line 161~234)  

#### 4) Generalization of training data  

 The hardest part to meet the requiment of thie project was to pass curves which have dirt field on right side. Vehicle can not detect border from paved road and dirt field. The generalization of color detection was supposed as driver for this difficulty.  
 Then, color composition shuffle part was implemented into generator of training data, to make the model be more generalized.  
(line 192~195)  

**Fig. the hardest part of driving corse**  

![alt text][image8]  
