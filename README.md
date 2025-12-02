# RGB相机基础捕获示例：camera_capture
支持USB相机、RTSP流、CSI相机(Jetson)
## USB相机
python camera_capture.py --source 0 --width 1280 --height 720 --fps 30

## RTSP网络相机
python camera_capture.py --source "rtsp://192.168.1.100:554/stream"

## Jetson CSI相机
python camera_capture.py --source "nvarguscamerasrc" --width 1920 --height 1080

# 单目相机标定示例：chessboard_calibration
## 步骤1: 采集标定图像(移动棋盘格到不同位置和角度)
python chessboard_calibration.py --mode capture --camera 0

## 步骤2: 执行标定计算
python chessboard_calibration.py --mode calibrate --images "calib_images/*.jpg"

## 步骤3: 测试畸变校正效果
python chessboard_calibration.py --mode test --camera 0
