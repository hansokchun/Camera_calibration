import cv2 as cv
import numpy as np

video_file = "c:/Users/benet/Desktop/vision/vision3/chessboard.mp4"
output_file = "c:/Users/benet/Desktop/vision/vision3/chessboard_correction.mp4"

K = np.array([[1999.53955, 0, 980.500564],
              [0, 2028.57661, 557.305516],
              [0, 0, 1]])
dist_coeff = np.array([0.2575275921259914, 0.038343559866068064, -0.0011735632043227467, 0.003449879732682275, -6.324789728470607])

video = cv.VideoCapture(video_file)

frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv.CAP_PROP_FPS))

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    valid, img = video.read()
    if not valid:
        break
    
    undistorted_img = cv.undistort(img, K, dist_coeff)
    
    out.write(undistorted_img)
    
    cv.imshow('Rectified', undistorted_img)
    if cv.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

video.release()
out.release()
cv.destroyAllWindows()
