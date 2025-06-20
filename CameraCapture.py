import cv2
import os 

#Right or Left
cam_name = "Left"

if cam_name == "Right":
    cam_port = "/dev/video2" #Use your own port
    cam_dir = "Right"
elif cam_name == "Left":
    cam_port = "/dev/video4"
    cam_dir = "Left"

cam = cv2.VideoCapture(cam_port)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
imgCount = 0

if not os.path.exists("images"):
    os.makedirs("images")
    print("images folder created!")
else:
    print("Images folder already exists.")

while True:
    _, image = cam.read()
    a, corner = cam.read()
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
 
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        cv2.drawChessboardCorners(corner, (9,6), corners2, ret)
    
    combined = cv2.hconcat([image, corner])
    cv2.imshow(cam_name, combined)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
    elif key == ord('c'):
        cv2.imwrite(f"images/{cam_dir}{imgCount}.png", image)
        imgCount += 1
        print(f'{imgCount} image captured.')

cv2.destroyWindow('image') 
