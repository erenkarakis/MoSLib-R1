import cv2 as cv
import numpy as np
import glob

# Yüklediğiniz kalibrasyon dosyalarındaki parametreleri alın
mtxL = np.load('Values/LeftCamMatrix.npy')
distL = np.load('Values/LeftDistortionCoeffs.npy')
mtxR = np.load('Values/RightCamMatrix.npy')
distR = np.load('Values/RightDistortionCoeffs.npy')

# Chessboard boyutları
board_size = (9, 6)  # 9x6 şablon

# Gerçek dünyadaki objenin noktaları (düz zemin üzerinde 3D koordinatları)
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

# Görüntü noktalarını depolayacak liste
objpoints = []  # 3D nokta
imgpointsL = []  # Sol kameradan 2D noktalar
imgpointsR = []  # Sağ kameradan 2D noktalar

# Görüntülerin yolunu tanımla
imagesL = sorted(glob.glob("stereoimg/Left*.png"), key=lambda x: int(x.split('/')[-1].split('Left')[-1].split('.')[0]))
imagesR = sorted(glob.glob("stereoimg/Right*.png"), key=lambda x: int(x.split('/')[-1].split('Right')[-1].split('.')[0]))

for fnameL, fnameR in zip(imagesL, imagesR):
    imgL = cv.imread(fnameL)
    imgR = cv.imread(fnameR)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    retL, cornersL = cv.findChessboardCorners(grayL, board_size, None)
    retR, cornersR = cv.findChessboardCorners(grayR, board_size, None)

    if retL and retR:
        objpoints.append(objp)
        corners2L = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners2R = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        imgpointsL.append(corners2L)
        imgpointsR.append(corners2R)

        cv.drawChessboardCorners(imgL, board_size, corners2L, retL)
        cv.drawChessboardCorners(imgR, board_size, corners2R, retR)

        cv.imshow('Left Image', imgL)
        cv.imshow('Right Image', imgR)
        cv.waitKey(300)

cv.destroyAllWindows()

# Stereo kalibrasyonu yap
ret, mtxL, distL, mtxR, distR, R, T, R1, R2 = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1],
    flags=cv.CALIB_FIX_INTRINSIC, criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
)

# Çıktıları kontrol et
print(f"R shape: {R.shape}")
print(f"T shape: {T.shape}")
print(f"R1 shape: {R1.shape}")
print(f"R2 shape: {R2.shape}")


# Çıktıları kontrol et
print(f"R shape: {R.shape}")
print(f"T shape: {T.shape}")
print(f"R1 shape: {R1.shape}")

# Yeni matrisleri kaydet
np.save('Values/R.npy', R)
np.save('Values/R1.npy', R1)
np.save('Values/R2.npy', R2)
np.save('Values/T.npy', T)

