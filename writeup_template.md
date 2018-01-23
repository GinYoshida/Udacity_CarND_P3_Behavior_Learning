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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted 

#### 1. Submission of codes and report

My project includes the following files:
* "Code_test.ipynb" containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_tf_50.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results (This file)

#### 2. Submission of binary files
With following command, acutonomous driving is possible with simulator.
```sh
python drive.py model_tf_50.h5
```
As evidence, recorded video is included in this repository as "run1.mp4".

### Model Architecture and Training Strategy

#### 3. Solution Design Approach

##### Basic strategy
 Based on training material in this project of Udacity and following blog, transfer learning with lower level feature for this project was applied.
 Because, "small data set" and "Different data set" were seems to be this takes' type.

 This is Link of blog, which shows transfer learning types.
 https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2

 GoogleNet, InceptionV3 in Keras "Applications", was applied with only lower level feature, i.e. until Mixed1 layter.
 
 This is link for Keras document.
 https://keras.io/ja/applications/#inceptionv3

 Model architecture is shown as below.
 
 High parameters for training is:
 
 

##### Solution for each key point
---
 Following points are difficulty to reach the goal.
1) Data augmentation data

 The driving record was created based on keyborad input only. With this operation, almost input of steering angle is 25 or 0. To ajudst from very discrete data to continuous data, I added final column with the following operation.

 If |(Moving average of steering angle with 5 rows)| > |steering angle|, apply moving average. Another case is opposite way.

 Original steering data distiburion is shown in Fig xxx.

 Then, data argumentataion was done with refering NVIDIA approach.
 xxxx
 Steering angle adjustment from the left/right camera was 0.2.
 Shifting 

https://arxiv.org/pdf/1604.07316.pdf

2) Memory handling with argumented training data
 To handle.
 

3) Generalization
 The most time consuming part to solving
 Color shuffle of compositon of color ois


 Over fitting
 With Adam, Batch nomralization and Dropout, not 

![alt text][image3]
![alt text][image4]
![alt text][image5]

