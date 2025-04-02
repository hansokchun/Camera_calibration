import cv2 as cv
import numpy as np
import glob

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    video = cv.VideoCapture(video_file)
    img_select = []
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, _ = cv.findChessboardCorners(gray, board_pattern)
        if found:
            img_select.append(frame)
        if not select_all and len(img_select) > 20:
            break
    video.release()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = np.array(obj_pts, dtype=np.float32) * board_cellsize
    obj_points_list = []
    
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
            obj_points_list.append(obj_points)
    
    assert len(img_points) > 0, 'No valid chessboard images found!'
    
    ret, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(obj_points_list, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)
    
    return ret, K, dist_coeff

video_file = "c:/Users/benet/Desktop/vision/vision3/chessboard.mp4"
board_pattern = (10, 7)  
board_cellsize = 25.0  

images = select_img_from_video(video_file, board_pattern)

rms, K, dist_coeff = calib_camera_from_chessboard(images, board_pattern, board_cellsize)

print("## Camera Calibration Results")
print(f"* The number of applied images = {len(images)}")
print(f"* RMS error = {rms}")
print("* Camera matrix (K) =")
print(K)
print("* Distortion coefficient (k1, k2, p1, p2, k3, ...) =")
print(dist_coeff.flatten().tolist())