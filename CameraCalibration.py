import numpy as np
import cv2 as cv
import glob
import os
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Witch cam will be calibrate Left or Right
camera = "Left"

images = glob.glob(f'images/{camera}*.*')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('rmse:', ret)
print('camera matrix:\n', mtx)
print('distortion coeffs:', dist)
print(f"fx: {mtx[0, 0]}")
print(f"fy: {mtx[1, 1]}")

if not os.path.exists("Values"):
    os.makedirs("Values")
    print("Values folder created!")
else:
    print("Values folder already exists.")

# Saving values
np.save(f'Values/{camera}CamMatrix', mtx)
np.save(f'Values/{camera}DistortionCoeffs', dist)
 
cv.destroyAllWindows()