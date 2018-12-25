
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1_output.jpg "Undistorted"
[image2]: ./output_images/test2_undistorted.jpg. "Road Transformed"
[image3]: ./output_images/test2_binary_threshold.jpg "Binary Example"
[image4]: ./output_images/test2_perspectiveTransform.jpg "Warp Example"
[image5]: ./output_images/test2_fitPolynomial.jpg "Fit Visual"
[image6]: ./output_images/test2_lanesDrawn.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"


### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

Using the mtx and dist from camera calibration obtained as above, test image is undistorted by using function `cv2.undistort()`.

![alt text][image2]

#### 2. Color and gradient thresholding

I used a combination of color and gradient thresholds to generate a binary image. For this purpose, L and S channel of HLS color space is used. L channel is used for the gradient threshold and S channel is used for the color threshold. For the gradient threshold, I have used the combination of x and y gradient along with the direction of the gradient.

combined_binary[(s_binary == 1) | ((sxbinary == 1) & (sybinary == 1) & (dir_binary==1))] = 1

Here s_binary is the color threshold in the S channel, sxbinary and sybinary are the gradient threshold in the L channel, dir_binary is the direction oof gradient threshold.

![alt text][image3]

#### 3. Perspective transform

To perform the perspective transform, we need to set the axis value in the region of interest. I have selected 4 points on the line which appears to be straight. I have
used `./test_images/straight_lines1.jpg` for this purpose.  4 points manually chosen are:

    y1_left=y2_right=720
    y2_left=y1_right=455
    x1_left=190
    x2_left=585
    x1_right=705
    x2_right=1130
src = np.float32([[x2_left,y2_left],[x1_right,y1_right],[x2_right,y2_right],[x1_left,y1_left]])

For these src points, we need to specify dst points where we want to transform the image.

dst = np.float32([
        [offset, 0],
        [img_size[0]-offset, 0],
        [img_size[0]-offset, img_size[1]], 
        [offset, img_size[1]]
    ])
Here I have set an offset value of 100 to get the dst points. img_size is (720,1280).
Using these src and dst points, we can calculate the perspective transform matrix using cv2.getPerspectiveTransform().
Once we have the perspective transform matrix M, we can transform the image using these matrix. The lines in the warped image should 
be vertical since I have used straight line.

These same src and dst points can be used for the other images where lines are curved to get the warped image. The lines in the 
warped image in this case will be curved.


![alt text][image4]

#### 4. Identify the lane pixel
I have used `get_lane_pixels()` function which takes the binary warped image and returns the lane pixels of the left and right line.
First step is to take a histogram of the bottom half of the image.
Calculate the peak of the left and right halves of the histogram which denotes the starting point of the lanes. We need to set the number of sliding windows and (+/-) margin of the window in which we need to look for the pixel.
Following are the hyperparameters.
    nwindows = 9
    margin = 100
    minpix = 50
    window_height = np.int(binary_warped.shape[0]//nwindows) 
Iterate through the windows in both left and right lane and get the nonzero pixels identified in the window region in both the lanes.
This is done by following code:
good_left_inds = ((win_xleft_low <= nonzerox) & (nonzerox< win_xleft_high) &
        (nonzeroy<win_y_high) &((win_y_low <= nonzeroy))).nonzero()[0]
        good_right_inds = ((win_xright_low <= nonzerox) & (nonzerox< win_xright_high) &
        (nonzeroy<win_y_high) &((win_y_low <= nonzeroy))).nonzero()[0]

For each window append these indices to the left lane and right lane indices list.
If the pixels in the window are found to be less than minimum pixels then recentre the window using mean of pixels in the window.

I have also used `effective_search()` function to look for the pixels in the margin of the fitted polynomial function. Using this I 
can quickly search for the lines instead of manually applying window technique which becomes ineffective if we use it on all the frames.
 
![alt text][image5]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I have used `measure_curvature_real()` function to calculate the radius of the curvature of the lane. This function returns the left
and right radius of curvature. I have taken the mean of the left and right radius of curvature.
To convert the radius of the curvature from pixel space to the real world space, I have used 
ym_per_pix = 30/720 
xm_per_pix = 3.7/700
    
To compute the distance of vehicle with respect to center, I have used the following code:
    dist_from_centre=(img.shape[1]/2)*xm_per_pix - (right_line.line_base_pos+left_line.line_base_pos)/2
  where line_base_pos is the x value of the fitted line at the bottom of the image.


#### 6. Draw the Lane area.

For the first frame, we need to detect the lanes using window technique. Once the left and right lines are detected, we can use
quick search around the previous fitted line to get the left and right line pixel in the next frame. I have used `effective_search()` to search around the polynomial. For this purpose I have set a margin of 80 pixel to search around the polynomial.
Use the average value of polynomial coefficients and average x value of the last n fitted lines to smoothen the line acroos the frames.

Following is the code to quick search around the poynomial. I have used the best fit (average over the last n fits)
    leftx, lefty, rightx, righty=effective_search(binary_warped,left_line.best_fit,right_line.best_fit)

I have used the following code to get the left and right lane pixels. Here left_line.bestx denotes the average x value of the last 
n fitted lines.
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

Now that I have left and right line pixels, to draw the lane onto the warped blank image I have used:
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

![alt text][image6]

---

### Pipeline (video)

I have created a Line Class which keeps track of lines detected among the recent frames like polynomial fit, average value of polynomial
fits and average value of pixels over the last n frames. Based on this information, the lines are detected for the next frame. 
`process()` function takes each frame and detects the line using `get_lane_pixels()` for the first frame and `effective_search()` for the 
next frames once the lines are detected. Finally it draws the lane area for each frame.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Shortcomings

The color space and thresholding needs to be finetuned. This pipeline does not work very well on the challenge problems.
I should also look for the other color channels other than HLS to see how it works.

#### 2. Next Plan

I will look for the other color channels which can adapt better in the different lighting conditions.

  
