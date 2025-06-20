import cv2
import numpy as np

#Kalibrasyon Verilerini Yükle
mtxL = np.load('Values/LeftCamMatrix.npy')
distL = np.load('Values/LeftDistortionCoeffs.npy')
mtxR = np.load('Values/RightCamMatrix.npy')
distR = np.load('Values/RightDistortionCoeffs.npy')
R = np.load('Values/R.npy')
T = np.load('Values/T.npy')

#Kameraları Başlat ve Çözünürlük Ayarla
width, height = 640, 480
capL = cv2.VideoCapture("/dev/video4")
capR = cv2.VideoCapture("/dev/video2")
capL.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, width)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#Stereo Rectify ve Map Hazırlığı
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtxL, distL, mtxR, distR, (width, height), R, T)
mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (width, height), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (width, height), cv2.CV_32FC1)

#Tıklama Fonksiyonu ve Globaller
clicked_point, depth_map = None, None
def on_mouse_click(event, x, y, flags, param):
    global clicked_point, depth_map
    if event == cv2.EVENT_LBUTTONDOWN and depth_map is not None:
        distance = depth_map[y, x]
        msg = f"{x}, {y}: {distance:.2f} m" if not np.isnan(distance) else f"{x}, {y}: NaN"
        print(msg)
        clicked_point = (x, y)

cv2.namedWindow("Disparity")
cv2.setMouseCallback("Disparity", on_mouse_click)

#Trackbarlar
cv2.createTrackbar("numDisparities (×16)", "Disparity", 6, 16, lambda x: None)
cv2.createTrackbar("blockSize", "Disparity", 7, 21, lambda x: None)
cv2.createTrackbar("Uniqueness", "Disparity", 10, 50, lambda x: None)
cv2.createTrackbar("Speckle Window", "Disparity", 100, 200, lambda x: None)
cv2.createTrackbar("Speckle Range", "Disparity", 15, 100, lambda x: None)

#Ana Döngü
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Kameralardan görüntü alınamadı.")
        break

    #Düzleştirme (Rectify)
    rectL = cv2.remap(frameL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapRx, mapRy, cv2.INTER_LINEAR)

    grayL, grayR = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    #Trackbar Değerlerini Oku
    numDisp = cv2.getTrackbarPos("numDisparities (×16)", "Disparity") * 16
    blockSize = cv2.getTrackbarPos("blockSize", "Disparity") | 1  # Tek sayı yap
    uniqueness = cv2.getTrackbarPos("Uniqueness", "Disparity")
    speckleWindow = cv2.getTrackbarPos("Speckle Window", "Disparity")
    speckleRange = cv2.getTrackbarPos("Speckle Range", "Disparity")

    #Stereo Eşleştirici Ayarları
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=max(16, numDisp),
        blockSize=blockSize,
        P1=8 * 1 * blockSize ** 2,
        P2=32 * 1 * blockSize ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness,
        speckleWindowSize=speckleWindow,
        speckleRange=speckleRange
    )

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan  # Geçersiz noktalar

    #Derinlik Haritası Hesaplama
    focal_length = P1[0, 0]
    baseline = 0.1  # T vektörünün uzunluğu
    depth_map = (focal_length * baseline) / disparity

    #Disparity Görselleştirme
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_vis = cv2.GaussianBlur(disp_vis, (7, 7), 0)
    disp_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    if clicked_point:
        x, y = clicked_point
        d = depth_map[y, x]
        label = f"{d:.2f} m" if not np.isnan(d) else "NaN"
        cv2.circle(disp_colormap, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(disp_colormap, label, (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    #Ekrana Yazdır
    cv2.imshow("combined", cv2.hconcat([rectL, rectR]))
    cv2.imshow("Disparity", disp_colormap)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC ile çık
        break

#Kapatma
capL.release()
capR.release()
cv2.destroyAllWindows()
