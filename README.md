# COMS4995- Kaggle-Project

Team Members:
* Ashwathy Mohan Menon (am5683)
* Soham Dandapath (sd3596)
* Suvansh Dutta (sd3513)

While the traditional image classification task deals with classifying a single object in an image, in this competition we focus on hierarchical labeling of an entity in the image. In our hierarchical classification task, an image belongs to a superclass and subclass. In this paper, we implemented and analyzed three approaches: computing direct probabilities, fine-tuning a CNN with multiple heads, and an encoder-decoder architecture. We found the multi-head CNN to perform the best on the held-out test set. We also exploited attention layers of the decoder to interpret the model. <br>

This repository contains the released data and code for all the three methods listed in the report.

* Independent Probability Approach
    * Convolution Transpose 2D
    * Residual Block Like structure
    * Contrastive Learning
* CNN Multi-Head Model
* Encoder-Decoder Architecture
  * CNN-RNN 
  * CNN-Transformers

# Data 
The data is split in 70-30 stratified ratio of train and test and can be found in the data_split folder. Each image is a 8x8 pixels. 

# Folder Structure 
* CNN Multi Head Model  : CNN\_MultiHead\_Model.ipynb 
* Independent Probability Approach : 
    * Convolution Transpose 2D : Superclass Conv2DTranspose.ipynb
    * Residual Block Like structure : Superclass Skip Connection.ipynb
    * Contrastive Learning : Superclass Contrastive.ipynb

# Results 
We obtained the following results <br/>
<img width="536" alt="Screenshot 2022-12-19 at 11 47 16 AM" src="https://user-images.githubusercontent.com/42071654/208477134-be04dbac-4e2d-4cbc-a04d-59055a31ee09.png">

