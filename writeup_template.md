**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training.png "MSE"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 93). I used the ELU activation rather than RELU based on advise from my reviewer after the traffic signs project. Apparently it's faster.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 105, 108, 110). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 58). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I didn't need very much recovery data since the additional camera data from either of the car was included to simulate the car being off course.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was implement the Nvida model from class with a couple of extra tweaks. One planned tweak was to use a better colorspace and the other was to include some dropout layers to avoid overfitting.

My first step was to use a convolution neural network model similar to the Nvida model which seemed appropriate since it was designed for this very task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that drop out layers were included before flattening the convolution output and before the final two dense layers. This ensured the the model was robust and didn't rely heavily on any single local or global feature. 

Then I experimented with different values for the number of epochs, the dropout rate and the batch size. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases I recorded some extra training data with the car travelling smoothly around the problem sections. I also used the mouse over the keyboard for steering to ensure smoother values that reflected how a user normally drives a car.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 93-113) consisted of a convolution neural network with the following layers and layer sizes

-Data normalization node to project data into \[-1, 1] range with mean centered at 0
-Image cropping node to remove parts of the image that are not useful for driving (sky etc)
-Added a 1x1x3 convolution with a linear activate to try and learn the best colorspace. I read about this in a medium post discussing the traffic signs project.
-Following are layers taken from the nVidia model, the back to back convolution nets add plenty of non linearity to the model
- Convolution 5x5x24, subsample=(2,2), elu activation
- Convolution 5x5x36, subsample=(2,2), elu activation
- Convolution 5x5x48, subsample=(2,2), elu activation
- Convolution 5x5x64, elu activation
- Convolution 5x5x64, elu activation
- Dropout
- Flatten
- Dense, 100, activation=elu
- Dropout
- Dense, 50, activation=elu
- Dropout
- Dense, 10, activation=elu
- Dense, 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to know what action to take if it found itself off the normal driving path for which it was otherwise trained. I didn't need to record much of this since the additional cameras automatically provide this simulation.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would provide more training data with less manual collection.

I also used the cameras on the left and right side of the car as training examples, except I replaced their recorded steering angle with one that indicated the correction the car should make to correct from its incorrect course. I experimented with different correction angles to come to the one in my code which gave the best results (model.py 19).

After the collection process, I had 106722 data points. I didn't do any pre-processing of the data other than the normalizing, mean centering and cropping that occured in the keras model.

To improve performance I only stored the textual information describing the training data, along with a parameter to determine if the image was to be flipped. Later during training a generator would be called for each batch and only at that time would the image actually be read into memory and processed to flip if appropriate.

I finally randomly shuffled the data set and put 0.2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15 as evidenced by the MSE loss of the training and validation data converging and leveling out as seen below. I also included an automatic early stopping component in the network but I found if I let it train for that many epochs the model overfitted and didn't generalize well. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![MSE][image1]
