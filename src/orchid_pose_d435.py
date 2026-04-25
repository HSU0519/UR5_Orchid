import time
import cv2
import math
import numpy as np
from ultralytics import YOLO

from angle import *
from orchid_seg import orchid_seg_predict_block, orchid_seg_leafs_number_predict_block2


# ==================== 通用工具函式 ====================

def distance(point1, point2):
    """計算兩點之間的歐幾里得距離"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def check_point_to_points(point1, points, point_l, point_r):
    """檢查 point1 與 points 中每個點的距離，回傳最近且在範圍內的葉片資訊"""
    dis_l = distance(point1, point_l)
    dis_r = distance(point1, point_r)
    for point_info in points:
        pt = point_info[0]
        if distance(point1, pt) <= dis_l and distance(point1, pt) <= dis_r:
            return [point_info[1], point_info[2]]
    return [0, 0]


def orchid_RGB_modified(img, dot):
    """檢查指定座標周圍 3×3 範圍是否存在特定顏色，回傳對應分級值 (2~5) 或 0"""
    diameter = 3
    height, width, _ = img.shape
    x, y = dot[0], dot[1]

    color_map = {
        (56, 56, 255): 5,      # 紅色
        (151, 157, 255): 4,    # 綠色
        (49, 210, 207): 3,     # 黃綠色
        (10, 249, 72): 2,      # 青綠色
    }

    for i in range(-(diameter // 2), (diameter // 2) + 1):
        for j in range(-(diameter // 2), (diameter // 2) + 1):
            cx, cy = x + i, y + j
            if 0 <= cx < width and 0 <= cy < height:
                bgr = tuple(img[cy, cx])
                if bgr in color_map:
                    return color_map[bgr]
    return 0


def orchid_RGB(img, dot):
    """檢查指定座標周圍 3×3 範圍是否全為黑色 (用於判斷土壤區域)"""
    diameter = 3
    height, width, _ = img.shape
    x, y = dot[0], dot[1]

    for i in range(-(diameter // 2), (diameter // 2) + 1):
        for j in range(-(diameter // 2), (diameter // 2) + 1):
            cx, cy = x + i, y + j
            if 0 <= cx < width and 0 <= cy < height:
                b, g, r = img[cy, cx]
                if b > 0 or g > 0 or r > 0:
                    return False
    return True


def hsv2bgr(h, s, v):
    """HSV 轉 BGR (數值範圍 0~255)"""
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb_map = {
        0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
        3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
    }
    r, g, b = rgb_map.get(h_i, (0, 0, 0))
    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    """根據 ID 產生隨機顏色 (用於視覺化)"""
    h = (((id << 2) ^ 0x937151) % 100) / 100.0
    s = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h, s, 1)


# ==================== 共用常數 ====================

# 5 關鍵點模型的骨架連接
SKELETON_5KPT = [[2, 1], [3, 1], [4, 1], [5, 1]]

# 3 關鍵點模型 (病徵) 的骨架連接: up-center, down-center
SKELETON_3KPT = [[1, 2], [3, 2]]

# 姿態調色盤
POSE_PALETTE = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
    [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
    [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
], dtype=np.uint8)

# 5 關鍵點顏色索引
KPT_COLOR_5 = POSE_PALETTE[[10, 0, 9, 7, 16]]
# 3 關鍵點顏色索引 (病徵模型)
KPT_COLOR_3 = POSE_PALETTE[[10, 0, 9]]

# 骨架線顏色
LIMB_COLOR = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
LIMB_COLOR_3 = POSE_PALETTE[[9, 7]]


# ==================== 共用繪圖函式 ====================

def draw_keypoints(img, keypoint, kpt_color):
    """在影像上繪製關鍵點"""
    for i, (x, y, conf) in enumerate(keypoint):
        color_k = [int(c) for c in kpt_color[i]]
        cv2.circle(img, (int(x), int(y)), 5, color_k, -1, lineType=cv2.LINE_AA)


def draw_skeleton(img, keypoint, skeleton, limb_color):
    """在影像上繪製骨架線"""
    for i, sk in enumerate(skeleton):
        pos1 = (int(keypoint[sk[0] - 1, 0]), int(keypoint[sk[0] - 1, 1]))
        pos2 = (int(keypoint[sk[1] - 1, 0]), int(keypoint[sk[1] - 1, 1]))
        conf1 = keypoint[sk[0] - 1, 2]
        conf2 = keypoint[sk[1] - 1, 2]
        if conf1 < 0.5 or conf2 < 0.5:
            continue
        if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
            continue
        cv2.line(img, pos1, pos2, [int(c) for c in limb_color[i]], thickness=2, lineType=cv2.LINE_AA)


def draw_angle_info(img, nrow, left, top, right, angle_soil, angle_rotate):
    """在影像上繪製角度相關的線段與文字"""
    x0, y0 = nrow[1]
    x1, y1 = nrow[2]
    x2, y2 = nrow[3]
    x3, y3 = nrow[4]
    x4, y4 = nrow[5]
    center_x = (left + right) // 2
    center_y = (top + int(nrow[1][1])) // 2  # 用 bbox 與 kpt0 的中間

    cv2.line(img, (x3, y3), (x4, y4), (255, 0, 0), 2)       # 土壤線
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)      # 旋轉線
    cv2.line(img, (left, y0), (right, y0), (0, 0, 255), 2)    # 水平參考線
    cv2.putText(img, f'{int(angle_soil)} {int(angle_rotate)} deg',
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ==================== 共用解析函式 ====================

def parse_keypoints_5(keypoint):
    """解析 5 關鍵點模型的結果，回傳 row (CSV用) 和 nrow (座標用)"""
    row_kpts = []
    nrow = []
    for i, (x, y, conf) in enumerate(keypoint):
        row_kpts.append(f"({int(x)}, {int(y)})")
        nrow.append([int(x), int(y)])
    return row_kpts, nrow

def safe_depth(depth_frame, x, y):
    h, w = depth_frame.shape[:2]
    return float(depth_frame[np.clip(y, 0, h-1), np.clip(x, 0, w-1)])

def compute_soil_info(nrow_a, nrow_b, depth_frame):
    """計算土壤兩點的質心、距離 (含深度)"""
    cx = int((nrow_a[0] + nrow_b[0]) / 2)
    cy = int((nrow_a[1] + nrow_b[1]) / 2)
    dist = math.sqrt(
        (nrow_a[0] - nrow_b[0]) ** 2 +
        (nrow_a[1] - nrow_b[1]) ** 2 +
        (safe_depth(depth_frame, nrow_a[0], nrow_a[1]) - safe_depth(depth_frame, nrow_b[0], nrow_b[1])) ** 2
    )
    return cx, cy, dist


# ==================== 預測函式 ====================

def orchid_pose_predict_d435(img, depth_frame, pose_model_name, predict_pose_number):
    """
    單純 Pose 預測 (無分割)，使用 5 關鍵點模型。
    kpt0=中心, kpt1=左土壤, kpt2=右土壤, kpt3=左葉, kpt4=右葉
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name)
    img_name = f"predict-pose-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True, imgsz=640)[0]
    exetime = time.time() - start_time

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]  # kpt0 = 中心

        draw_keypoints(img, keypoint, KPT_COLOR_5)

        # 計算土壤角度 (kpt1, kpt2) 與旋轉角度 (kpt3, kpt4)
        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[1], nrow[2], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            # 繪製角度資訊 (需要帶 id 的 nrow，前面補 id)
            nrow_with_id = [id] + nrow
            draw_angle_info(img, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img, csv_data, img_name, exetime, exetime2


def orchid_pose_seg_area_predict_d435(img, depth_frame, pose_model_name, seg_model_name, predict_pose_number):
    """
    Pose + 分割預測 (無葉片計數)。
    透過分割結果判斷土壤關鍵點是否在黑色區域 (土壤)。
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name)
    img_name = f"predict-pose-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    # 語義分割
    start_time_3 = time.time()
    img_seg, img_seg_name = orchid_seg_predict_block(img, seg_model_name, predict_pose_number)
    exetime3 = time.time() - start_time_3
    img_seg_copy = img_seg.copy()

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]

        draw_keypoints(img, keypoint, KPT_COLOR_5)

        # 確認土壤關鍵點在黑色區域
        if not (orchid_RGB(img_seg_copy, nrow[1]) and orchid_RGB(img_seg_copy, nrow[2])):
            continue

        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[1], nrow[2], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            nrow_with_id = [id] + nrow
            draw_angle_info(img, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_seg_area_leafs_number_predict_d435(img, depth_frame, pose_model_name, seg_model_name, predict_pose_number):
    """
    Pose + 分割 + 葉片計數預測 (舊版 5 關鍵點模型)。
    kpt1/kpt2 為土壤, kpt3/kpt4 為葉片。
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name, task='pose')
    img_name = f"predict-pose-seg-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    # 語義分割 + 葉片計數
    start_time_3 = time.time()
    img_seg, img_seg_name, results_row_leafs_seg = orchid_seg_leafs_number_predict_block2(
        img, seg_model_name, predict_pose_number)
    exetime3 = time.time() - start_time_3
    img_seg_copy = img_seg.copy()

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)",
                 "Leafs-Number", "Leafs-Area", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]

        draw_keypoints(img_seg, keypoint, KPT_COLOR_5)

        # 確認土壤關鍵點在黑色區域
        if not (orchid_RGB(img_seg_copy, nrow[1]) and orchid_RGB(img_seg_copy, nrow[2])):
            continue

        # 葉片資訊
        needed_leafs = check_point_to_points(nrow[0], results_row_leafs_seg, nrow[3], nrow[4])
        results_rows.append(needed_leafs)
        row.append(needed_leafs[1])  # 葉片數
        row.append(needed_leafs[0])  # 葉片面積

        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[1], nrow[2], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            nrow_with_id = [id] + nrow
            draw_angle_info(img_seg, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img_seg, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img_seg, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_predict_d435_new(img, depth_frame, pose_model_name, predict_pose_number):
    """
    新版 Pose 預測 (無分割)，使用 5 關鍵點模型。
    與舊版差異：kpt3/kpt4 為土壤, kpt1/kpt2 為旋轉。
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name)
    img_name = f"predict-pose-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]

        draw_keypoints(img, keypoint, KPT_COLOR_5)

        # 新版：kpt3/kpt4 為土壤, kpt1/kpt2 為旋轉
        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[3], nrow[4], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            nrow_with_id = [id] + nrow
            draw_angle_info(img, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img, csv_data, img_name, exetime, exetime2


def orchid_pose_seg_area_predict_d435_new(img, depth_frame, pose_model_name, seg_model_name, predict_pose_number):
    """
    新版 Pose + 分割預測 (無葉片計數)。
    kpt3/kpt4 為土壤, kpt1/kpt2 為旋轉。
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name)
    img_name = f"predict-pose-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    # 語義分割
    start_time_3 = time.time()
    img_seg, img_seg_name = orchid_seg_predict_block(img, seg_model_name, predict_pose_number)
    exetime3 = time.time() - start_time_3
    img_seg_copy = img_seg.copy()

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]

        draw_keypoints(img, keypoint, KPT_COLOR_5)

        # 確認土壤關鍵點在黑色區域 (新版用 kpt3/kpt4)
        if not (orchid_RGB(img_seg_copy, nrow[3]) and orchid_RGB(img_seg_copy, nrow[4])):
            continue

        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[3], nrow[4], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            nrow_with_id = [id] + nrow
            draw_angle_info(img, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_seg_area_leafs_number_predict_d435_new(img, depth_frame, pose_model_name, seg_model_name, predict_pose_number):
    """
    新版 Pose + 分割 + 葉片計數預測。
    kpt3/kpt4 為土壤, kpt1/kpt2 為旋轉參考。
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name, task='pose')
    img_name = f"predict-pose-seg-single{predict_pose_number}.jpg"

    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    img_copy = img.copy()

    # 語義分割 + 葉片計數
    start_time_3 = time.time()
    img_seg, img_seg_name, results_row_leafs_seg = orchid_seg_leafs_number_predict_block2(
        img, seg_model_name, predict_pose_number)
    exetime3 = time.time() - start_time_3
    img_seg_copy = img_seg.copy()

    if results.boxes.data.tolist() is None or results.boxes.id is None:
        return None, img, None, img_name, exetime, 0

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(results.boxes.id.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(x0, y0)", "(x1, y1)", "(x2, y2)", "(x3, y3)", "(x4, y4)",
                 "Leafs-Number", "Leafs-Area", "Soil-Angle", "Rotate-Angle"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        row_kpts, nrow = parse_keypoints_5(keypoint)
        row = [id] + row_kpts
        results_rows = [id, [nrow[0][0], nrow[0][1]]]

        draw_keypoints(img_copy, keypoint, KPT_COLOR_5)

        # 確認土壤關鍵點在黑色區域 (kpt3/kpt4)
        if not (orchid_RGB(img_seg_copy, nrow[3]) and orchid_RGB(img_seg_copy, nrow[4])):
            continue

        # 葉片資訊
        needed_leafs = check_point_to_points(nrow[0], results_row_leafs_seg, nrow[1], nrow[2])
        results_rows.append(needed_leafs)
        row.append(needed_leafs[1])
        row.append(needed_leafs[0])

        angle_soil = int(calculate_angle_soil(left, top, right, bottom,
                                              nrow[3][0], nrow[3][1], nrow[4][0], nrow[4][1]))
        angle_rotate = int(calculate_angle_rotate(left, top, right, bottom,
                                                  nrow[1][0], nrow[1][1], nrow[2][0], nrow[2][1])) % 180

        cx, cy, soil_dist = compute_soil_info(nrow[3], nrow[4], depth_frame)
        results_rows.append([angle_soil, [cx, cy], soil_dist])
        row.append(angle_soil)

        if angle_rotate > 90:
            angle_rotate -= 180
        row.append(angle_rotate)

        if angle_rotate != 0 and abs(angle_rotate) >= 10:
            results_rows.append(int(angle_rotate))
            ALL_results_rows.append(results_rows)
            nrow_with_id = [id] + nrow
            draw_angle_info(img_copy, nrow_with_id, left, top, right, angle_soil, angle_rotate)

        csv_data.append(row)
        draw_skeleton(img_copy, keypoint, SKELETON_5KPT, LIMB_COLOR)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img_copy, csv_data, img_name, exetime, exetime2, exetime3


def orchid_pose_sick_predict_d435(img, depth_frame, pose_model_name, seg_model_name, predict_pose_number):
    """
    病徵蘭花專用預測函式 (3 關鍵點模型)。
    kpt0=上方夾取點, kpt1=植株中心, kpt2=下方夾取點。
    使用 results.plot() 繪製 bounding box、關鍵點與骨架線。

    回傳格式:
    results_rows = [id, [center_x, center_y], [leafs_area, leafs_number],
                    [grip_angle, [grip_centroid_x, grip_centroid_y], grip_distance]]
    """
    ALL_results_rows = []
    model = YOLO(pose_model_name, task='pose')
    img_name = f"predict-pose-sick-{predict_pose_number}.jpg"

    # YOLO 預測
    start_time = time.time()
    results = model.track(source=img, verbose=False, device=0, conf=0.25, iou=0.45,
                          save=False, tracker="bytetrack.yaml", persist=True)[0]
    exetime = time.time() - start_time

    # 用 plot() 繪製

    ids_backup = results.boxes.id if results.boxes.id is not None else None
    img_copy = results.plot(line_width=1) if results.boxes is not None and len(results.boxes) > 0 else img.copy()

    # 語義分割 + 葉片計數
    start_time_3 = time.time()
    img_seg, img_seg_name, results_row_leafs_seg = orchid_seg_leafs_number_predict_block2(
        img, seg_model_name, predict_pose_number)
    exetime3 = time.time() - start_time_3

    # 檢查是否有有效偵測
    if results.boxes.data.tolist() is None or ids_backup is None:
        return None, img, None, img_name, exetime, 0, exetime3

    names = results.names
    boxes = results.boxes.data.tolist()
    ids = np.array(ids_backup.cpu(), dtype="int")
    keypoints = results.keypoints.cpu().numpy()

    csv_data = [["id", "(up_x, up_y)", "(center_x, center_y)", "(down_x, down_y)",
                 "Leafs-Number", "Leafs-Area", "Grip-Angle", "Label"]]

    start_time_2 = time.time()

    for obj, keypoint, id in zip(boxes, keypoints.data, ids):
        left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
        confidence = obj[4]
        label = int(obj[5])

        # 解析 3 關鍵點: nrow[0]=up, nrow[1]=center, nrow[2]=down
        row = [id]
        nrow = []
        for i, (x, y, conf) in enumerate(keypoint):
            row.append(f"({int(x)}, {int(y)})")
            nrow.append([int(x), int(y)])

        # 定位點 = center
        results_rows = [id, [nrow[1][0], nrow[1][1]]]

        # 葉片資訊
        needed_leafs = check_point_to_points(nrow[1], results_row_leafs_seg, nrow[0], nrow[2])
        results_rows.append(needed_leafs)
        row.append(needed_leafs[1])
        row.append(needed_leafs[0])

        # 夾取角度與距離 (up → down)
        up_x, up_y = nrow[0]
        down_x, down_y = nrow[2]

        grip_angle = int(calculate_angle_soil(left, top, right, bottom,
                                              up_x, up_y, down_x, down_y))
        grip_cx = int((up_x + down_x) / 2)
        grip_cy = int((up_y + down_y) / 2)
        grip_dist = math.sqrt(
            (up_x - down_x) ** 2 +
            (up_y - down_y) ** 2 +
            (safe_depth(depth_frame, up_x, up_y) - safe_depth(depth_frame, down_x, down_y)) ** 2
        )

        row.append(grip_angle)
        row.append(names[label])
        results_rows.append([grip_angle, [grip_cx, grip_cy], grip_dist])

        ALL_results_rows.append(results_rows)
        csv_data.append(row)

    exetime2 = time.time() - start_time_2
    return ALL_results_rows, img_copy, csv_data, img_name, exetime, exetime2, exetime3