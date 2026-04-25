# ============================================================
# Eye-to-Hand 校正程式（PnP 4x4 矩陣版 - 單張照片免變高度）
#
# 流程：
#   1. 探針點棋盤格 4 個角落 → pose.txt（擷取 X, Y, Z）
#   2. 用 3D 向量展開 132 個機器人三維世界座標 (XYZ)
#   3. 用 cv2.findChessboardCorners 偵測單張照片的像素座標
#   4. 使用 cv2.solvePnP 配合相機內部參數，解出 6 自由度姿態
#   5. 轉換出最終 4x4 C44 Eye-to-Hand 矩陣
# ============================================================

import sys
import os

# 解決 mpl_toolkits 衝突
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]
for mod in list(sys.modules.keys()):
    if 'mpl_toolkits' in mod:
        del sys.modules[mod]

import cv2
import numpy as np

# ==================== 設定區 ====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 自動搜尋 data 資料夾中最後一個包含 A 和 D 子資料夾的目錄
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

if not os.path.exists(DATA_ROOT):
    raise ValueError(f"找不到資料夾: {DATA_ROOT}")

data_dirs = sorted([
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
    and os.path.isdir(os.path.join(DATA_ROOT, d, 'A'))
    and os.path.isdir(os.path.join(DATA_ROOT, d, 'D'))
])

if not data_dirs:
    raise ValueError(f"在 {DATA_ROOT} 中找不到包含 A 和 D 子資料夾的目錄")

DATA_DIR = os.path.join(DATA_ROOT, data_dirs[-1])

# 自動尋找 A 資料夾內的第一張圖片
a_dir_path = os.path.join(DATA_DIR, 'A')
img_files = sorted([f for f in os.listdir(a_dir_path) if f.endswith(('.png', '.jpg'))])
if not img_files:
    raise ValueError(f"在 {a_dir_path} 中找不到任何圖片")
IMG_NAME = img_files[4]

print(f"設定輸入資料目錄: {DATA_DIR}")
print(f"設定輸入校正影像: {IMG_NAME}")

# 棋盤格設定
BOARD_SIZE = (11, 12)  # X方向角點數, Y方向角點數
COLS, ROWS = 11, 12
POINT_NUM = ROWS * COLS
SQUARE_SIZE_MM = 25.0  # 格子尺寸(mm)

# 相機內部參數 (請依 RealSense 實際狀況微調)
FX, FY = 615.0, 615.0
CX, CY = 320.0, 240.0

# ================================================

def main():
    print("="*60)
    print("  Eye-to-Hand 校正啟動 (純 PnP 4x4 矩陣)")
    print("="*60)

    # ---------------- 步驟 1 & 2 ----------------
    pose_file = os.path.join(DATA_DIR, 'pose.txt')
    print(f"\n[步驟] 讀取探針姿態: {pose_file}")
    
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"找不到 {pose_file}")
        
    raw_pose = np.loadtxt(pose_file)
    corners_pose = raw_pose[:4]

    # 取出 4 個角落 (轉換為毫米 mm)
    P_TL = corners_pose[0][:3] * 1000  # 左上
    P_TR = corners_pose[1][:3] * 1000  # 右上
    P_BL = corners_pose[2][:3] * 1000  # 左下
    P_BR = corners_pose[3][:3] * 1000  # 右下

    # 求出平面的 X 與 Y 單位方向向量 (平均抵銷探針誤差)
    X_dir = ((P_TR - P_TL) + (P_BR - P_BL)) / 2.0
    Y_dir = ((P_BL - P_TL) + (P_BR - P_TR)) / 2.0
    X_unit = X_dir / np.linalg.norm(X_dir)
    Y_unit = Y_dir / np.linalg.norm(Y_dir)

    # 產生 132 個 3D 空間點
    world_XYZ = np.zeros((POINT_NUM, 3), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            idx = r * COLS + c
            world_XYZ[idx] = P_TL + c * SQUARE_SIZE_MM * X_unit + r * SQUARE_SIZE_MM * Y_unit

    print(f"  └─ 成功建立 132 個機器人 3D 座標 (Z 會自動具備傾角)")

    # ---------------- 步驟 3 ----------------
    img_path = os.path.join(DATA_DIR, 'A', IMG_NAME)
    print(f"\n[步驟] 讀取影像進行特徵擷取: {IMG_NAME}")
    
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"無法讀取影像 {img_path}")

    ret, corners = cv2.findChessboardCorners(img, BOARD_SIZE)
    if not ret:
        raise ValueError(f"這張照片未找到棋盤格")

    # 亞像素精化 (讓邊緣定位到小數點級別)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    image_points = corners_refined.reshape(-1, 2)

    # OpenCV 的找點順序通常是從下到上，所以要做順序反轉
    image_points = image_points[::-1].copy().astype(np.float32)

    print(f"  └─ 成功偵測到 {len(image_points)} 個精準像素角點")

    # ---------------- 步驟 4 ----------------
    print(f"\n[步驟] 執行 SolvePnP 計算 4x4 矩陣")
    camera_matrix = np.array([
        [FX,  0, CX],
        [ 0, FY, CY],
        [ 0,  0,  1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))

    success, rvec, tvec = cv2.solvePnP(world_XYZ, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("PnP 演算法解算失敗！")

    # 旋轉向量 (3x1) 轉換為旋轉矩陣 (3x3)
    R_r2c, _ = cv2.Rodrigues(rvec)

    # 組裝 4x4 轉換矩陣 (Robot to Camera)
    T_r2c = np.eye(4)
    T_r2c[:3, :3] = R_r2c
    T_r2c[:3, 3] = tvec.flatten()

    # 取逆矩陣 => (Camera to Robot) => 最終使用的 C44 矩陣！
    C44_eyetohand = np.linalg.inv(T_r2c)

    print("\n✅ 計算成功！ 4x4 C44_eyetohand 矩陣：")
    print("  C44_EYETOHAND = np.array([")
    for i, row in enumerate(C44_eyetohand):
        comma = "," if i < 3 else ""
        print(f"      [{row[0]:15.8e}, {row[1]:15.8e}, {row[2]:15.8e}, {row[3]:15.8e}]{comma}")
    print("  ])")

    # ---------------- 步驟 5 ----------------
    print(f"\n[步驟] 匯出結果檔")
    
    # 部署至 src/calibration_data 中
    deploy_dir = os.path.join(SCRIPT_DIR, 'calibration_data')
    os.makedirs(deploy_dir, exist_ok=True)
    deploy_path = os.path.join(deploy_dir, 'C44_eyetohand.npy')
    np.save(deploy_path, C44_eyetohand)
    print(f"  └─ 部署至: {deploy_path}")

    print("\n 校正程式執行完畢！\n")

if __name__ == "__main__":
    main()
