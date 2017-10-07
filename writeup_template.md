#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[training_barchart]: ./examples/training_barchart.png "Training Data Visualization"
[validation_barchart]: ./examples/validation_barchart.png "Validation Data Visualization"
[test_barchart]: ./examples/test_barchart.png "Training Test Visualization"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Below is the bar chart showing the number of samples for each unique classes in training data.

![alt text][training_barchart]


Below is the bar chart showing the number of samples for each unique classes in validation data.

![alt text][validation_barchart]


Below is the bar chart showing the number of samples for each unique classes in test data.

![alt text][test_barchart]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have done following three steps as part of pre-processing
1. Convert the images to grayscale because it would reduce the number of channels in the input image thus reducing the size of the network. Also the color of the traffic signal image is not that relevant in identifying the sign. 
2. After converting the image to gray scale, the pixel values range from 0 to 255. So to normalize the values, I used the formula pixel = (pixel - 128)/128. It really helped the network to learn much better. Since all the weights and bias values are close to 0 with normal distribution, it helps that input data is also normalized so that values are close to 0 with normal distribution.
3. As last step, I had to reshape the input data to (32x32x1) to fit into LeNet network. When applying the grayscale conversion the shape of input changed from (32x32x3) to (32x32)

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x8 				|
| Convolution 4x4	    | 1x1 stride, valid padding, outputs 10x10x18			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x18 				|
| Flatten          		| Output 450        									|
| Fully connected		  | Input 450, Output 150        									|
| RELU					|												|
| Dropout					|		With keep_prob = 0.5										|
| Fully connected		  | Input 150, Output 100        									|
| RELU					|												|
| Dropout					|	With keep_prob = 0.5				|
| Fully connected		  | Input 100, Output 43        									|
| Softmax				|         									|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I started with standard LeNet 5 architecture with following characteristics.
 1. AdamOptimizer as the optimizer
 2. Batch size as 128
 3. Number of Epochs as 10.
 4. Learning rate as 0.001

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.958
* test set accuracy of 0.932

Following are the steps I have taken to achieve my final results.

1. First I started with LeNet architecture that was used for MNIST dataset in the previous chapter as I felt it is good starting point for traffic sign recongition task.
2. I applied the traffic sign data to LeNet architecture as it is just by modifying the input and output size to get initial learning rate. I found the learning rate to be ~86 which was pretty good start.
3. I have done data pre-processing by converting the input images to grayscale and also normalizing the input by using the foruma pixel = (pixel-128)/128. This steps gave me pretty good results and my learning rate improved beyong 90%
4. Then I tried playing around with increasing epoch size, reducing learning rate and also increasing the network size without modifying the architecture much. I was stuck in this step for quite sometime and my learning rate didn't go beyond 92.
5. I went to the forums and read what are the things other people are trying and I found link to below paper. I glanced through the paper and took some inputs from it. 
 http://people.idsia.ch/~juergen/nn2012traffic.pdf
6. Specifically I changed the following three things.
    a. Changed size of first convolution from 5x5 to 7x7
    b. Changed size of second convolution from 5x5 to 4x4
    c. Added dropout after each Fully connected layer. (I felt this made huge impact. I think it was probably because lot of images in the input are not very clear so dropping out some of neurons in the networks helped it to ignore them and learn better)
7. The above changes significantly improved my learning rate and I could go beyond 94.
8. I also tuned the depth of different layers to get optimal learning rate. I found that generally increasing the depth of layers increased my learning rate. But I stuck with some optimal values so as not to increase them too much for little gain. 
9. I found help from the forums to plot the training and validation loss on graph to monitor how my network is learning. I found that useful to tune my learning rate and number of epochs.
10. Finally I reached a point where I was able to get accuracy beyond 95 and I was satisfied.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


