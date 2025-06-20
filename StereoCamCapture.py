import cv2
import os

L = cv2.VideoCapture("/dev/video4")
R = cv2.VideoCapture("/dev/video2")

if not os.path.exists("stereoimg"):
    os.makedirs("stereoimg")
    print("stereoimg folder created!")
else:
    print("stereoimg folder already exists.")

imgCount = 0
cv2.waitKey(5000)
while True:
    if not (L.grab() and R.grab()):
        print("No more frames")
        break
       
    a, left = L.retrieve()
    b, right = R.retrieve()

    combined = cv2.hconcat([left, right])
    cv2.imshow("Stereo View", combined)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
    elif key == ord('c'):
        cv2.imwrite(f"stereoimg/Left{imgCount}.png", left)
        cv2.imwrite(f"stereoimg/Right{imgCount}.png", right)
        imgCount += 1
        print(f'{imgCount} image captured.')

L.release()
R.release()
cv2.destroyAllWindows() 