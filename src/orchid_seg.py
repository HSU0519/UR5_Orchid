import cv2
import numpy as np
from hex2rgb import hex2rgb
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def calculate_area_and_centroid(points):
    """
    使用 OpenCV 計算多邊形的面積和幾何中心（質心）
    :param points: 多邊形頂點的列表，每個頂點由 (x, y) 坐標表示
    :return: (面積, 幾何中心坐標 (cx, cy))
    """
    # 將頂點數組轉換為 OpenCV 可接受的格式
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    # 計算多邊形的面積
    area = cv2.contourArea(contour)
    
    # 計算輪廓的矩
    M = cv2.moments(contour)
    
    # 計算質心坐標
    if M["m00"] != 0:  # 確保面積不為零，避免除零錯誤
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # 如果面積為零，則無法計算質心
        raise ValueError("多邊形的面積為零，無法計算質心")
    
    return [cx, cy], area
    

def orchid_seg_predict(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']

    Yatsen = ['五片', '四片', '土下', '土上', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
    
    frame_PIL = Image.fromarray(img)
    draw = ImageDraw.Draw(frame_PIL, "RGBA")
    text_font = ImageFont.truetype('Arial_Unicode_MS.ttf', 30)
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return None, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            color = hex2rgb(HEX[class_id % 20])
            xy = [tuple(point) for point in seg]

            if len(xy) >= 2:
                draw.polygon(xy, fill=(color[0], color[1], color[2], 128))

            # draw.rectangle((x, y, x2, y2), outline=color, width=5)
            # left, top, right, bottom = draw.textbbox((x + 5, y - 37), f'{objname} {score}', font=text_font)
            # draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill=color)
            # draw.text((x + 5, y - 37), f'{objname} {score}', fill='white', font=text_font)
        
        return np.array(frame_PIL), img_name
    
    
def orchid_seg_predict_block(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    # Yatsen = ['五片', '四片', '土下', '土上', '三片', '二片']
    
    Yatsen = ['葉子', '土']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.45, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return None, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            # color = hex2rgb(HEX[class_id % 20])
            xy = np.array([point for point in seg])
            
            if class_id == 0:
                color = [0, 255, 0]
            
            else:
                color = [0, 0, 0]

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name
    
    
def orchid_seg_leafs_number_predict_block(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    Yatsen = ['五片', '四片', '土', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.5, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []

    if (results[0].masks) is None:
        return img, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            color = hex2rgb(HEX[class_id % 20])
            xy = np.array([point for point in seg])
            
            if class_id == 2:
                color = [0, 0, 0]

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name
    
    
def orchid_seg_leafs_number_predict_block2(img, seg_model_name, predict_pose_number):
    
    HEX = ['FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
           '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7']
    
    Yatsen = ['五片', '四片', '土', '三片', '二片']
    
    names_language = Yatsen
    
    model = YOLO(seg_model_name, task='segment')
    img_name = "predict-seg-single" + str(predict_pose_number) + ".jpg"
        
    results = model.predict(source = img, verbose = False, device = 0, conf = 0.25, iou = 0.5, 
                            save = False)
    
    
    height, width, channels = img.shape

    segmentations = []; results_row = [];

    if (results[0].masks) is None:
        return img, None

    else:
        for seg in results[0].masks.xyn:
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentations.append(segment)

        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        scores = np.array(results[0].boxes.conf.cpu(), dtype="float").round(2)

        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            (x, y, x2, y2) = bbox
            objname = names_language[class_id]

            # color = hex2rgb(HEX[class_id % 20])
            color = [0, 255, 0]
            xy = np.array([point for point in seg])
            
            if len(xy) >= 2 and class_id == 0:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 5])
                
            elif len(xy) >= 2 and class_id == 1:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 4])
            
            elif class_id == 2:
                color = [0, 0, 0]
            
            elif len(xy) >= 2 and class_id == 3:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 3])
                
            elif len(xy) >= 2 and class_id == 4:
                leafs_centroid_seg, leafs_area = calculate_area_and_centroid(xy)
                results_row.append([leafs_centroid_seg, leafs_area, 2])

            if len(xy) >= 2:
                img = cv2.fillPoly(img, [xy], color=[color[0], color[1], color[2]])
        
        return img, img_name, results_row
                
                
                
