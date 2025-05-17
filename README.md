# CamShiftOD
This code implements the CamShift (Continuously Adaptive Mean Shift) algorithm using OpenCV in Python to track an object in a video

## The key concepts of this project:

HSV color space: More robust for color-based tracking than BGR.

Histogram backprojection: Finds pixels similar to the target object.

CamShift: An improved version of MeanShift that updates window size and angle dynamically.

cv.boxPoints(): Converts CamShift output to 4 corner points of the rotated rectangle.

