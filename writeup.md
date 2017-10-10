# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Find the code in `features.py`. This is called from `train.py` during the training process and from `detect.py` during the detection process.

`cars_notcars.py` loads the `cars` and `notcars` examples. They look something like this:

![example of car and notcar][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![hog example][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

All three of spatial features, histogram features and hog features helped improve accuracy. Using YCrCB performed very well. I tried changing orientations, pix_per_cell, and other values, but they seemed to not change performance much.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In `train.py` I trained a linear SVM using scaled versions of the features described above. On my 2016 Macbook my SVC can identify 7000 labels per second. The training and test data was randomized, and the test data composed of 20% of the total dataset, a.k.a. 1800 of 9000 images each of notcars and cars.

###Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

`detect.py` defines the `find_cars()` function which first extracts features for the entire image, then slides a window across a portion of the image to detect cars. `pipeline()` does this detection for window sizes of 1.5, 2, and 2.5 with a growing portion of the image (y pixels 400-550, then 400-600, then 350-650).

This results in a bunch of bounding boxes, which is filtered using a heatmap (see below).

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I improved speed of detection and reduced false positives by only searching a portion of the image where I knew cars would show up (a.k.a. not searching the sky).

To speed up my debugging cycles I built a realtime debug feature. This played back a video to the screen using pygame functions which allowed me to step frame by frame and reload the detection code. This let me look at problematic frames and quickly test a bunch of different methods. See `detect_movie_live.py` for the pygame code that enables this feature. I think it's pretty cool.

Using the live detection, I further improved my detection code by


 I also improved performance by ___ .

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

See `detect_with_labels.py` for my code to do this. I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

You can run `detect_movie_live.py` to see this in action.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I found it tricky to get the window sizes correct to detect cars in my images. At different sizes different cars were detected. I found a combination of window sizes by using my live video code to quickly test out different sizes.

I also added some of my own images of car/notcar sourced from the video. This risks overfitting but without it the cars were not always detected well.

There are some limitations with my current approach. It will fail to detect cars where they don't normally appear, like the sky. It globs together cars when they get too close. It also has small gaps on the right side of the image that don't have any windows because the windows sizes aren't all factors of the total width.

I could make this more robust by following a car from frame to frame and focusing the sliding window detection around a known car. I could also estimate direction and speed of the cars (at least in pixel values) to help predict where the cars will be in the next frames.
