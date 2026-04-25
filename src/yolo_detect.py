from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

# 載入 YOLO 模型
model = YOLO("/home/jen-lab/Desktop/UR5/models/best_sick_keypoint.v23i.yolo26s-BinaryAttention-rtmopose_opset19.onnx", task='pose')

# 建立並設定 RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 開始串流
profile = pipeline.start(config)

# 取得深度感測器的 Depth Scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# 建立對齊物件，將深度圖對齊到彩色圖
align_to = rs.stream.color  
align = rs.align(align_to)

# 建立深度圖破洞填補濾鏡
hole_filling = rs.hole_filling_filter()

try:
    print("開始即時檢測... 按下 'Esc' 或 'q' 鍵離開視窗。")
    while True:
        # 1. 獲取相機幀
        frames = pipeline.wait_for_frames()
        
        # 2. 將深度圖對齊彩色圖
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 驗證幀是否有效
        if not aligned_depth_frame or not color_frame:
            continue

        # 3. 處理深度圖 (填補破洞)
        filled_depth = hole_filling.process(aligned_depth_frame)   
        depth_frame_modify = np.asanyarray(filled_depth.get_data())
        
        # 取得彩色影像的 Numpy 陣列
        color_image = np.asanyarray(color_frame.get_data())    

        # 4. YOLO 預測
        # 現在 task 已經正確設定，verbose=True 也不會再報 KeyError 了
        results = model(color_image, verbose=True, conf=0.25, iou=0.45)

        # 5. 繪製結果與影像拼接
        # results[0].plot() 會回傳畫好 bounding box 與關鍵點的影像
        annotated_image = results[0].plot()
        dc_images = np.hstack((annotated_image, color_image))
        
        # 顯示影像
        cv2.imshow("Real-Time YOLO Pose Detection", dc_images)

        # 6. 按鍵檢測 (1 毫秒延遲，維持即時更新)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # 按下 Esc 或 'q' 關閉
            break

finally:
    # 確保關閉相機與視窗釋放資源
    pipeline.stop()
    cv2.destroyAllWindows()
    print("程式已結束，資源已釋放。")