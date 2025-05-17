import numpy as np
import cv2 as cv

# Step 1: Load the video
cap = cv.VideoCapture('sample.mp4')
ret, frame = cap.read()

# Step 2: Define the initial tracking window (ROI)
x, y, width, height = 400, 440, 150, 150
track_window = (x, y, width, height)
roi = frame[y:y + height, x:x + width]

# Step 3: Convert ROI from BGR to HSV color space
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

# Step 4: Create a mask for filtering the desired color range
mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# Step 5: Compute the histogram for the hue channel in ROI
roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

# Step 6: Setup termination criteria: 15 iterations or move by at least 2 points
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 2)

# Step 7: Start tracking loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for consistent processing
    frame = cv.resize(frame, (720, 720), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    cv.imshow('Original', frame)

    # Optional preprocessing: Thresholding
    ret1, frame1 = cv.threshold(frame, 180, 155, cv.THRESH_TOZERO_INV)

    # Convert current frame to HSV
    hsv = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)

    # Step 8: Perform back projection using histogram
    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Step 9: Apply CamShift to get new tracking window
    ret2, track_window = cv.CamShift(dst, track_window, term_crit)

    # Step 10: Draw the tracking result (rotated rectangle)
    pts = cv.boxPoints(ret2)
    pts = np.int0(pts)
    result = cv.polylines(frame, [pts], True, (0, 255, 255), 2)

    # Show the tracking output
    cv.imshow('Camshift', result)

    # Exit condition: Press ESC (key code 27)
    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

# step 11: Release resources
cap.release()
cv.destroyAllWindows()
