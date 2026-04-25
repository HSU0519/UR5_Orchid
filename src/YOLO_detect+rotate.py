import sys
import os
import warnings
import traceback

# 過濾 math3d 棄用警告，避免淹沒真正的錯誤訊息
warnings.filterwarnings("ignore", message=".*deprecated.*dist_squared.*")

# 確保優先使用虛擬環境的 site-packages
venv_site_packages = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.venv', 'lib', 'python3.10', 'site-packages'
)
if os.path.exists(venv_site_packages):
    sys.path.insert(0, venv_site_packages)

import urx
from urx.urrobot import RobotException
import socket
import subprocess
import cv2
import time
from math import radians, degrees
import numpy as np
import pyrealsense2 as rs
from function_arm import rv2rpy
from algorithm.ant import create_distance_matrix, ant_colony_optimization
from orchid_pose_d435 import orchid_pose_seg_area_leafs_number_predict_d435_new, orchid_pose_sick_predict_d435

# ============================================================
# 操作說明：
#   q/Q : 回歸初始姿態
#   x/X : 拍照並紀錄當前座標與深度
#   r/R : 檢測一般蘭花並執行一次夾取路徑
#   p/P : 啟動自動連續辨識並夾取所有病徵蘭花
#   c/C : 執行 PnP 校正並更新轉換矩陣
#   b/B : 強制取消自動夾取模式  
#   j/k/l : 切換手動位移量 10mm / 1mm / 0.1mm
#   w/a/s/d/8/2 : 手動移動手臂
#   Esc : 結束程式
# ============================================================

# ==================== 常數設定 ====================

def load_calibration_matrix():
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration_data', 'C44_eyetohand.npy')
    if os.path.exists(calib_path):
        print(f"成功讀取校正矩陣: {calib_path}")
        return np.load(calib_path)
    else:
        print(f"警告：找不到校正矩陣 {calib_path}，將使用預設矩陣")
        return np.array([
            [-6.47790223e-01, -7.61747974e-01, -1.03851624e-02, -1.25343224e+02],
            [-7.61535471e-01,  6.47860893e-01, -1.84388279e-02, -4.64065017e+02],
            [ 2.07738803e-02, -4.03582290e-03, -9.99776054e-01,  1.06049421e+03],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ])

# D435i Eye-to-Hand PNP 4x4 轉換矩陣
C44_EYETOHAND = load_calibration_matrix()

# 預設保護深度，當讀取不到正確深度時使用的保守物理值 (mm)
DEFAULT_DEPTH_MM = 216.0

# 相機光學中心（近似，D435i 640×480 約為畫面中心）
CAM_CX = 320.0
CAM_CY = 240.0

OFFSET = {"x": 0.0, "y": 0.0, "z": -0.024, "angle": 45}

MODEL = {
    "POSE":  "models/best_all_0_degree_small.v2i.v11l_pose.pt",
    "SEG":   "models/best_Yat-sen_University_orchid-idea.v7i.v11s_seg.pt",
    "POSE2": "models/best_sick_keypoint.v23i.yolo26s-BinaryAttention-rtmopose_opset19.onnx",
}

UR5 = {"IP": "192.168.1.101", "PORT": 30001}

# 手臂運動參數
ACC = 0.1
VEL = 0.3
SAFE_Z = 0.484763          # 安全高度 (m)
PICK_Z = 0.29              # 夾取下降高度 (m)
PUT_Z_OFFSET = 0.06         # 苗架放置Z軸全域微調 (m)：如果您覺得放太深，改成正數(例: 0.01升高1公分)；放太淺改負數
DEFAULT_RX_RY_RZ = (0, 3.1271, 0)

# 苗架 4點插值陣列設定
def load_put_xyz():
    xyz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'put_xyz.txt')
    xyz_list = []
    if os.path.exists(xyz_path):
        with open(xyz_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) >= 4:
                # 永遠只取最後 4 次按鍵的紀錄：分別當作洞 1, 3, 13, 15
                recent_4 = lines[-4:]
                pts = [[float(p) for p in line.split(',')] for line in recent_4]
                P1, P3, P13, P15 = pts[0], pts[1], pts[2], pts[3]
                
                # 依序生成 3x5 共 15 個洞 (先 3 個用完，再跳下一行)
                for c in range(5):
                    for r in range(3):
                        # 計算比例
                        u = c / 4.0
                        v = r / 2.0
                        
                        # 雙線性插值 (Bilinear Interpolation)
                        top_x = P1[0] * (1 - u) + P13[0] * u
                        bot_x = P3[0] * (1 - u) + P15[0] * u
                        fx = top_x * (1 - v) + bot_x * v
                        
                        top_y = P1[1] * (1 - u) + P13[1] * u
                        bot_y = P3[1] * (1 - u) + P15[1] * u
                        fy = top_y * (1 - v) + bot_y * v
                        
                        top_z = P1[2] * (1 - u) + P13[2] * u
                        bot_z = P3[2] * (1 - u) + P15[2] * u
                        fz = top_z * (1 - v) + bot_z * v
                        
                        xyz_list.append([fx, fy, fz])
    return xyz_list

def load_put_counter():
    counter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'put_counter.txt')
    if os.path.exists(counter_path):
        try:
            with open(counter_path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0

def save_put_counter(counter):
    counter_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'put_counter.txt')
    os.makedirs(os.path.dirname(counter_path), exist_ok=True)
    with open(counter_path, 'w') as f:
        f.write(str(counter))

PUT_XYZ_LIST = load_put_xyz()
put_counter = load_put_counter()  # 紀錄連續放置的次數，重啟程式仍會保留

# 預備姿態 (joint angles, degrees)
HOME_JOINTS = [radians(a) for a in (-26.15, -71.91, -106.04, -91.31, 89.45, 63.88)]
DROP1_JOINTS = [radians(a) for a in (-26.15, -71.91, -88.07, -109.55, 89.45, 63.88)]
DROP2_JOINTS = [radians(a) for a in (-26.15, -71.91, -113.37, -84.91, 89.45, 63.88)]

CENTER_JOINTS = [radians(a) for a in (-4.88, -85.6, -58.91, -124.68, 89.26, 87.03)]

PIXEL_THRESHOLD = 40  # 重複偵測過濾閾值 (pixel)


# ==================== 工具函式 ====================

def filter_duplicate_poses(results_rows, threshold=PIXEL_THRESHOLD, log_prefix="蘭花"):
    """過濾像素距離過近的重複偵測"""
    if not results_rows:
        return results_rows

    filtered = []
    for row in results_rows:
        pt = np.array(row[1])
        if all(np.linalg.norm(pt - np.array(f[1])) >= threshold for f in filtered):
            filtered.append(row)

    print(f"{log_prefix} - YOLO 原始偵測數: {len(results_rows)}，過濾重複後有效夾持點: {len(filtered)}")
    return filtered


def send_gripper_script(script_path, replace_fn=None):
    """透過 socket 發送夾爪腳本至 UR5"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((UR5["IP"], UR5["PORT"]))

    with open(script_path) as f:
        lines = [replace_fn(line) if replace_fn else line for line in f]

    s.send(str.encode("".join(lines)))
    time.sleep(0.5)
    s.close()
    time.sleep(0.5)

    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))


def OPEN():
    send_gripper_script('src/open.txt')
    print('夾爪已開啟')


def CLOSE(width):
    def replace_pos(line):
        if "rq_set_pos_norm(100, " in line:
            pos = abs(round(width / 0.00085) - 100)
            return f'    rq_set_pos_norm({pos}, "1")\n'
        return line

    send_gripper_script('src/close.txt', replace_fn=replace_pos)
    print('夾爪已閉合')


def pixel_to_world(px, py, depth_image=None):
    """正統 PnP 4x4 空間座標轉換：像素 + 深度 → 相機 3D → 機器人世界 3D (m)"""
    # D435i 理論相機參數 (未來若有更準的 rs_intrinsics 可置換)
    fx, fy = 615.0, 615.0
    
    if depth_image is not None:
        # 深度圖數值轉為 float (此處的深度值預設即為 mm)
        z_mm = float(depth_image[int(py), int(px)])
    else:
        z_mm = DEFAULT_DEPTH_MM

    # 防止深度讀取黑洞 (0 或負數)
    if z_mm <= 0:
        z_mm = DEFAULT_DEPTH_MM

    # 反推相機座標系的三維物理座標 (Xc, Yc, Zc) 單位為 mm
    Xc = (px - CAM_CX) * z_mm / fx
    Yc = (py - CAM_CY) * z_mm / fy
    Zc = z_mm
    
    # 乘上正統 PnP 4x4 轉換矩陣
    camera_pt = np.array([Xc, Yc, Zc, 1.0], dtype=np.float32)
    robot_pt = C44_EYETOHAND @ camera_pt
    
    bx_mm = robot_pt[0]
    by_mm = robot_pt[1]
    bz_mm = robot_pt[2]
    
    # 返回 X 和 Y 和 Z（轉換回公尺給機器臂使用）
    return bx_mm / 1000.0, by_mm / 1000.0, bz_mm / 1000.0


def move_to_safe(bx, by):
    """移動到 (bx, by) 的安全高度"""
    rob.movel((bx + OFFSET["x"], by + OFFSET["y"], SAFE_Z, *DEFAULT_RX_RY_RZ), 1, 0.1)


def save_image_and_depth(color_img, depth_img, dir_name, idx):
    """儲存彩色影像與深度資料"""
    for sub in ("A", "D"):
        d = f"data/{dir_name}/{sub}"
        os.makedirs(d, exist_ok=True)

    cv2.imwrite(f"data/{dir_name}/A/A{idx}.png", color_img)
    with open(f"data/{dir_name}/D/D{idx}.txt", 'w') as f:
        for row in range(480):
            f.write(' '.join(f'{depth_img[row][col]:f}' for col in range(640)))
            f.write('\n')


def save_predict_images(color_copy, annotated_img, img_name, dir_name):
    """儲存原始與標註後的預測影像"""
    orig_dir = f"data/{dir_name}/Original"
    os.makedirs(orig_dir, exist_ok=True)
    cv2.imwrite(f"{orig_dir}/Orig_{img_name}", color_copy)
    cv2.imwrite(f"data/{dir_name}/{img_name}", annotated_img)

def draw_help_overlay(img, step_size):
    """在影像上繪製半透明操作面板"""
    overlay = img.copy()
    h, w = img.shape[:2]
    box_x0, box_x1 = w - 340, w - 10
    tx = box_x0 + 10
    cv2.rectangle(overlay, (box_x0, 10), (box_x1, 440), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    lines = [
        "=== UR5 Control Panel ===",
        "[Q] Home Pose",
        "[X] Record Data",
        "[R] Rotate Orchid",
        "[P] Pick Sick Orchid",
        "[C] Run Calibration",
        "[B] Cancel Auto Pick",
        "[0] Reset Put Counter",
        "[Z] Record Put XYZ Hole",
        "[W/S/A/D/8/2] Move X/Y/Z",
        "[J/K/L] Step: 10/1/0.1 mm",
        "[ESC] Exit",
        "-------------------------",
        f"Step: {step_size*1000:.1f} mm",
        f"Puts: {put_counter}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (tx, 35 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

# ==================== 夾取流程 ====================

def execute_normal_pick(best_route, annotated_img, depth_image):
    """執行旋轉蘭花的夾取路徑 (按 r)"""
    for i, loc in enumerate(best_route):
        bx, by, bz = pixel_to_world(loc[1][0], loc[1][1], depth_image)

        # 繪製路徑線
        if i > 0:
            cv2.line(annotated_img, tuple(best_route[i-1][1]), tuple(loc[1]), (255, 0, 255), 2)

        move_to_safe(bx, by)

        grip_align_angle = loc[3][0] + 90 + OFFSET["angle"]
        grip_align_angle = ((grip_align_angle + 180) % 360) - 180
        grip_width = (loc[3][2] + 12) / 1000

        if grip_align_angle == 0:
            continue

        # 開夾爪 → 旋轉對齊 → 下降 → 微調旋轉 → 上升
        CLOSE(grip_width)

        posej = rob.getj()
        if grip_align_angle < 0:
            posej[5] += radians(abs(grip_align_angle))
        else:
            posej[5] -= radians(abs(grip_align_angle))
        rob.movej(tuple(posej), ACC, VEL)

        posel_R = rob.getl()
        rob.movel((bx + OFFSET["x"], by + OFFSET["y"], PICK_Z + OFFSET["z"],
                   posel_R[3], posel_R[4], posel_R[5]), 1, 0.1)

        # 微調旋轉角度 (loc[4])
        posej = rob.getj()
        if loc[4] < 0:
            posej[5] -= radians(abs(loc[4]))
        else:
            posej[5] += radians(abs(loc[4]))
        rob.movej(tuple(posej), ACC, VEL)

        # 上升回安全高度
        posel_R2 = rob.getl()
        rob.movel((bx + OFFSET["x"], by + OFFSET["y"], SAFE_Z,
                   posel_R2[3], posel_R2[4], posel_R2[5]), 1, 0.1)
        rob.movel((bx + OFFSET["x"], by + OFFSET["y"], SAFE_Z, *DEFAULT_RX_RY_RZ), 1, 0.1)

    # 回歸預備姿態
    rob.movej(HOME_JOINTS, ACC, VEL)



def execute_sick_pick(best_route, annotated_img, depth_image):
    """執行病徵蘭花的夾取流程 (按 p 自動模式)"""
    global put_counter
    global PUT_XYZ_LIST
    for i, loc in enumerate(best_route):
        bx, by, bz = pixel_to_world(loc[1][0], loc[1][1], depth_image)

        # 取得根部偏移量
        offset_x, offset_y = 0.0, 0.0
        if len(loc) >= 4 and len(loc[3]) >= 4:
            down_px, down_py = loc[3][3][0], loc[3][3][1]
            root_bx, root_by, root_bz = pixel_to_world(down_px, down_py, depth_image)
            offset_x = root_bx - bx
            offset_y = root_by - by

        if i > 0:
            cv2.line(annotated_img, tuple(best_route[i-1][1]), tuple(loc[1]), (255, 0, 255), 2)

        move_to_safe(bx, by)

        grip_align_angle = loc[3][0] + 90 + OFFSET["angle"]
        grip_align_angle = ((grip_align_angle + 180) % 360) - 180
        grip_width = (loc[3][2] + 12) / 1000

        # 預開夾爪
        CLOSE(grip_width)
        time.sleep(0.5)

        # 旋轉對齊
        posej = rob.getj()
        if grip_align_angle < 0:
            posej[5] += radians(abs(grip_align_angle))
        elif grip_align_angle > 0:
            posej[5] -= radians(abs(grip_align_angle))
        rob.movej(tuple(posej), ACC, VEL)

        # 下降插入
        posel_R = rob.getl()
        rob.movel((bx + OFFSET["x"], by + OFFSET["y"], PICK_Z + OFFSET["z"],
                   posel_R[3], posel_R[4], posel_R[5]), 1, 0.1)

        # 夾緊
        CLOSE(grip_width * 0.00001)

        # 垂直上拔 (維持夾爪目前的姿態)
        posel_R_after_pick = rob.getl()
        rob.movel((bx + OFFSET["x"], by + OFFSET["y"], SAFE_Z, posel_R_after_pick[3], posel_R_after_pick[4], posel_R_after_pick[5]), 1, 0.1)

        # 先移動到 CENTER_JOINTS 中繼點
        rob.movej(CENTER_JOINTS, ACC, VEL)

        # 動態位移至收集點放置 (維持旋轉，防止植物甩尾)
        if len(PUT_XYZ_LIST) > 0:
            put_idx = put_counter % len(PUT_XYZ_LIST)
            put_xyz = PUT_XYZ_LIST[put_idx]

            target_x = put_xyz[0] - offset_x
            target_y = put_xyz[1] - offset_y
            target_z = put_xyz[2] + PUT_Z_OFFSET

            # 第一段：平移到苗架上方 (維持 SAFE_Z)
            rob.movel((target_x, target_y, SAFE_Z, posel_R_after_pick[3], posel_R_after_pick[4], posel_R_after_pick[5]), ACC, VEL)
            # 第二段：垂直降到置放點
            rob.movel((target_x, target_y, target_z, posel_R_after_pick[3], posel_R_after_pick[4], posel_R_after_pick[5]), 1, 0.1)

            CLOSE(grip_width * 2) # 放開夾爪

            # 第三段：開爪後先垂直拔回安全高度
            rob.movel((target_x, target_y, SAFE_Z, posel_R_after_pick[3], posel_R_after_pick[4], posel_R_after_pick[5]), 1, 0.1)

            # 放置完成後，計數器加一
            put_counter += 1
            if put_counter >= 15:
                put_counter = 0
            save_put_counter(put_counter)

        else:
            print("警告：尚未記錄任何 XYZ 放置點！")

    # 全數完成後回歸預備姿態
    rob.movej(HOME_JOINTS, ACC, VEL)

# ==================== 主程式 ====================

# 初始化機器手臂
rob = urx.Robot(UR5["IP"])
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))

pose = rob.getl()
step_size = 10 / 1000  # 預設手動位移量

# 建立 RealSense 影像管道
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"深度比例 (Depth Scale): {depth_scale}")

align = rs.align(rs.stream.color)
hole_filling = rs.hole_filling_filter()

# 資料儲存目錄
DIR_NAME = time.strftime("%Y%m%d_%H%M%S")
os.makedirs(f"data/{DIR_NAME}", exist_ok=True)

photo_counter = 1
predict_pose_number = 1
auto_pick_mode = False

try:
    with open(f"data/{DIR_NAME}/pose.txt", 'w') as pose_file:
        while True:
            # 取得對齊後的影像
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            key = cv2.waitKeyEx(10)

            # 深度處理
            filled = hole_filling.process(depth_frame)
            depth_image = np.asanyarray(filled.get_data())
            colorized_depth = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            color_image = np.uint8(np.asanyarray(color_frame.get_data()))
            color_image_copy = color_image.copy()

            draw_help_overlay(colorized_depth, step_size)
            cv2.imshow('image and depth', np.hstack((color_image, colorized_depth)))

            k = key & 0xFF

            # ---------- 結束 ----------
            if k == 27:
                break

            # ---------- 歸零計數器 ----------
            if k == ord('0'):
                put_counter = 0
                save_put_counter(put_counter)
                print("計數器已歸零！")

            # ---------- 紀錄放苗孔 4端點 XYZ (Z) ----------
            if k in (ord('z'), ord('Z')):
                xyz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'put_xyz.txt')
                os.makedirs(os.path.dirname(xyz_path), exist_ok=True)
                posel_z = rob.getl()
                
                # 讀取現有內容並加入新點位
                lines = []
                if os.path.exists(xyz_path):
                    with open(xyz_path, 'r') as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                
                lines.append(f"{posel_z[0]},{posel_z[1]},{posel_z[2]}")
                
                # 只保留最後 4 行
                if len(lines) > 4:
                    lines = lines[-4:]
                
                # 寫回檔案
                with open(xyz_path, 'w') as f:
                    for line in lines:
                        f.write(line + '\n')
                
                count = len(lines)
                
                print(f"已紀錄動作 (檔案總計 {count} 點)。請確保最後4點順序為: 1號 -> 3號 -> 13號 -> 15號")
                
                if count >= 4:
                    PUT_XYZ_LIST = load_put_xyz()
                    print(f"成功讀取最後4點插值，完美生成 {len(PUT_XYZ_LIST)} 個陣列孔位！")

            # ---------- 手動位移量切換 ----------
            if k in (ord('j'), ord('J')):
                step_size = 10 / 1000
                print("位移量: 10mm")
            elif k in (ord('k'), ord('K')):
                step_size = 1 / 1000
                print("位移量: 1mm")
            elif k in (ord('l'), ord('L')):
                step_size = 0.1 / 1000
                print("位移量: 0.1mm")

            # ---------- 手動移動 ----------
            move_map = {
                ord('8'): (0, 0, +1),
                ord('2'): (0, 0, -1),
                ord('w'): (0, -1, 0),
                ord('s'): (0, +1, 0),
                ord('d'): (-1, 0, 0),
                ord('a'): (+1, 0, 0),
            }
            if k in move_map:
                dx, dy, dz = move_map[k]
                pose[0] += dx * step_size
                pose[1] += dy * step_size
                pose[2] += dz * step_size
                try:
                    rob.movel((pose[0], pose[1], pose[2], *DEFAULT_RX_RY_RZ), 1, 0.1)
                except RobotException:
                    pass

            # ---------- 回歸預備姿態 ----------
            if k in (ord('q'), ord('Q')):
                rob.movej(HOME_JOINTS, ACC, VEL)
                pose = rob.getl()
                print("已回歸預備姿態")

            # ---------- 拍照紀錄 ----------
            if k in (ord('x'), ord('X')):
                posel = rob.getl()
                save_image_and_depth(color_image, depth_image, DIR_NAME, photo_counter)

                cv2.imshow('AD', np.hstack((color_image, colorized_depth)))

                # 寫入姿態資料
                rpy = rv2rpy(posel[3], posel[4], posel[5])
                pose_file.write(' '.join(f'{v:f}' for v in [
                    posel[0], posel[1], posel[2],
                    rpy[0], rpy[1], rpy[2],
                    posel[3], posel[4], posel[5]
                ]) + '\n')
                pose_file.flush()

                print(f"已儲存 #{photo_counter}  "
                      f"X:{posel[0]*1000:.2f} Y:{posel[1]*1000:.2f} Z:{posel[2]*1000-400:.2f}")
                photo_counter += 1

            # ---------- 旋轉蘭花偵測與夾取 (r) ----------
            if k == ord('r') or k == ord('R'):
                ALL_results, annotated, csv_data, img_name, t_pred, t_angle, t_seg = \
                    orchid_pose_seg_area_leafs_number_predict_d435_new(
                        color_image, depth_image, MODEL["POSE"], MODEL["SEG"], predict_pose_number)

                ALL_results = filter_duplicate_poses(ALL_results, PIXEL_THRESHOLD, "旋轉蘭花")

                if not ALL_results:
                    print("未偵測到有效的蘭花姿態資料。")
                else:
                    csv_data += [[], ["Predict-time(sec):", t_pred],
                                 ["Angle-time(sec):", t_angle],
                                 ["Segment-time(sec):", t_seg]]

                    dist_matrix = create_distance_matrix(ALL_results)
                    route_idx, best_dist, t_ant = ant_colony_optimization(dist_matrix)
                    csv_data.append(["Ant Path-execute-time(sec):", t_ant])

                    best_route = [ALL_results[i] for i in route_idx]
                    save_predict_images(color_image_copy, annotated, img_name, DIR_NAME)
                    execute_normal_pick(best_route, annotated, depth_image)
                    predict_pose_number += 1

            # ---------- 自動夾取模式開關 ----------
            if k in (ord('p'), ord('P')):
                print("\n===== 啟動自動連續夾取病徵蘭花模式 =====\n")
                auto_pick_mode = True

            # ---------- 取消自動夾取模式 (原本為 c，現在改為 b) ----------
            if k in (ord('b'), ord('B')):
                print("\n===== 已強制取消自動夾取模式 =====\n")
                auto_pick_mode = False

            # ---------- 執行 PnP 校正並即時更新矩陣 ----------
            if k in (ord('c'), ord('C')):
                print("\n===== 執行 PnP Eye-to-Hand 快速校正 =====")
                try:
                    calib_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pnp_calibration.py")
                    # 使用 subprocess 呼叫校正腳本
                    subprocess.run([sys.executable, calib_script], check=True)
                    
                    # 校正完畢後，重新讀取剛輸出的 npy 檔案到全域變數
                    C44_EYETOHAND = load_calibration_matrix()
                    print("===== 校正完畢，已動態載入最新轉換矩陣 =====\n")
                except Exception as e:
                    print(f"校正執行失敗: {e}\n")

            # ---------- 自動夾取病徵蘭花 ----------
            if auto_pick_mode:
                ALL_results, annotated, csv_data, img_name, t_pred, t_angle, t_seg = \
                    orchid_pose_sick_predict_d435(
                        color_image, depth_image, MODEL["POSE2"], MODEL["SEG"], predict_pose_number)

                ALL_results = filter_duplicate_poses(ALL_results, PIXEL_THRESHOLD, "病徵蘭花")

                if not ALL_results:
                    print("畫面上已無偵測到帶病徵蘭花")
                    auto_pick_mode = False
                else:
                    csv_data += [[], ["Predict-time(sec):", t_pred],
                                 ["Angle-time(sec):", t_angle],
                                 ["Segment-time(sec):", t_seg]]

                    dist_matrix = create_distance_matrix(ALL_results)
                    route_idx, best_dist, t_ant = ant_colony_optimization(dist_matrix)
                    csv_data.append(["Ant Path-execute-time(sec):", t_ant])

                    best_route = [ALL_results[i] for i in route_idx]
                    save_predict_images(color_image_copy, annotated, img_name, DIR_NAME)
                    execute_sick_pick(best_route, annotated, depth_image)
                    predict_pose_number += 1

except KeyboardInterrupt:
    print("\n使用者中斷程式 (Ctrl+C)")
except Exception:
    print("\n===== 程式異常中斷 =====")
    traceback.print_exc()
    print("========================\n")
finally:
    pipeline.stop()
    rob.close()
    cv2.destroyAllWindows()
    print("程式已結束，資源已釋放。")
    os._exit(0)