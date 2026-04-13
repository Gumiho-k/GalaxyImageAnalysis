
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_ellipse_in_image(image_path):
    """
    在圖像中查找、分割並擬合橢圓。
    此函數執行流程中的步驟 1-4：讀取、分割、輪廓提取、橢圓擬合。

    Args:
        image_path (str): 輸入圖像的路徑。

    Returns:
        tuple: (原始圖像, 擬合出的橢圓參數)。
               橢圓參數格式為 ((center_x, center_y), (axis_1, axis_2), angle)。
    """
    # 1. 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖像 {image_path}")
        return None, None

    # 轉換為灰度圖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 圖像分割 (對於理想圖像，使用簡單閾值)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 3. 查找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("在圖像中未找到輪廓。")
        return img, None

    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        print("輪廓點太少，無法擬合橢圓。")
        return img, None

    # 4. 擬合橢圓，獲取投影參數
    ellipse_params = cv2.fitEllipse(largest_contour)
    
    # 創建一個掩碼，只保留橢圓區域的像素
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, ellipse_params, 255, -1)
    galaxy_only = cv2.bitwise_and(img, img, mask=mask)

    return galaxy_only, ellipse_params


def calculate_physical_parameters(ellipse_params, q0=0.0):
    """
    從擬合的橢圓參數計算物理參數（傾角、位置角）。
    此函數對應流程中的步驟 5：參數推導與修正。

    Args:
        ellipse_params (tuple): cv2.fitEllipse返回的橢圓參數。
        q0 (float): 星系盤的內在軸比（厚度/直徑），用於修正。
                    q0=0.0 對應無限薄盤的經典模型。
                    對於真實螺旋星系，典型值約為 0.1-0.2。

    Returns:
        dict: 包含物理參數的字典。
    """
    (center), (axis_1, axis_2), position_angle_deg = ellipse_params

    # 確定長軸和短軸
    major_axis = max(axis_1, axis_2)
    minor_axis = min(axis_1, axis_2)

    if major_axis == 0:
        return None

    # 觀測到的軸比 (b/a)
    q_obs = minor_axis / major_axis

    # --- 核心物理模型 ---
    # 根據論文，我們從觀測軸比 q_obs 推算傾角 i。
    # 簡單模型 (無限薄盤, q0=0): cos(i) = q_obs
    # 考慮盤面厚度的修正模型: cos^2(i) = (q_obs^2 - q0^2) / (1 - q0^2)
    
    cos2_i_numerator = q_obs**2 - q0**2
    if cos2_i_numerator < 0:
        # 如果觀測到的比內在厚度還“圓”，物理上不可能，視為正面
        cos2_i_numerator = 0 
    
    cos2_i = cos2_i_numerator / (1 - q0**2) if (1 - q0**2) != 0 else cos2_i_numerator

    inclination_rad = np.arccos(np.sqrt(cos2_i))
    inclination_deg = np.rad2deg(inclination_rad)
    
    # 用於圖像拉伸的比例因子，即 1 / cos(i)
    stretch_factor = 1.0 / np.sqrt(cos2_i) if cos2_i > 0 else 1.0

    return {
        "center": center,
        "position_angle_deg": position_angle_deg,
        "inclination_deg": inclination_deg,
        "axis_ratio_obs": q_obs,
        "stretch_factor": stretch_factor,
    }


def deproject_galaxy(image, params):
    """
    根據物理參數，對圖像進行解投影變換。
    此函數對應流程中的步驟 6-7：構建變換矩陣和圖像翹曲。

    Args:
        image (numpy.ndarray): 輸入的星系圖像。
        params (dict): `calculate_physical_parameters` 返回的物理參數字典。

    Returns:
        numpy.ndarray: 解投影後的圖像。
    """
    center = params["center"]
    angle = params["position_angle_deg"]
    stretch = params["stretch_factor"]
    h, w = image.shape[:2]

    # 6. 構建仿射變換矩陣
    # 數學核心：將解投影分解為一系列幾何變換
    # 變換順序: T_center * R_back * S * R_align * T_origin
    
    # 變換1: 將橢圓中心平移到坐標原點
    M_translate_to_origin = np.float32([[1, 0, -center[0]], [0, 1, -center[1]]])

    # 變換2: 旋轉，使橢圓長軸與x軸對齊
    M_rotate_align = cv2.getRotationMatrix2D(center, -angle, 1)

    # 變換3: 沿y軸（短軸方向）拉伸，使其變為圓形
    # 這一步是解投影的核心，scale_factor = 1/cos(i)
    M_stretch = np.array([[1, 0, 0], [0, stretch, 0]], dtype=np.float32)
    # 為了與其他2x3矩陣結合，我們需要一個完整的變換流程
    # 完整的變換矩陣組合
    # T_center * R_back * S * R_align * T_origin
    # OpenCV的warpAffine將這些操作整合了，我們需要構建最終的2x3矩陣 M
    
    # 構建從原點出發的完整變換
    # 先旋轉對齊
    M = cv2.getRotationMatrix2D((0,0), -angle, 1)
    # 再拉伸
    M[1,:] = M[1,:] * stretch
    # 再旋轉回去
    M_rot_back = cv2.getRotationMatrix2D((0,0), angle, 1)
    M = M_rot_back @ np.vstack([M, [0,0,1]])
    
    # 現在加入平移
    M[0, 2] += center[0] - (M[0, 0] * center[0] + M[0, 1] * center[1])
    M[1, 2] += center[1] - (M[1, 0] * center[0] + M[1, 1] * center[1])

    # 7. 應用仿射變換 (圖像翹曲)
    deprojected_img = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

    return deprojected_img


def main():
    """
    主執行函數
    """
    image_path = '_photo.png'

    # 步驟 1-4: 查找橢圓
    galaxy_img, ellipse_params = find_ellipse_in_image(image_path)
    if ellipse_params is None:
        return

    # 步驟 5: 計算物理參數 (這裡使用理想薄盤模型 q0=0)
    # 若要考慮厚度，可設 q0=0.15
    phys_params = calculate_physical_parameters(ellipse_params, q0=0.0)
    if phys_params is None:
        return

    print("從圖像推導出的物理參數:")
    print(f"  位置角 (PA): {phys_params['position_angle_deg']:.2f} 度")
    print(f"  觀測軸比 (b/a): {phys_params['axis_ratio_obs']:.3f}")
    print(f"  推算出的傾角 (i): {phys_params['inclination_deg']:.2f} 度 (假設 q0=0.0)")

    # 步驟 6-7: 解投影
    deprojected_img = deproject_galaxy(galaxy_img, phys_params)

    # 步驟 8: 保存並顯示結果
    output_path = 'find_shape.jpg'
    cv2.imwrite(output_path, deprojected_img)
    print(f"\n結果已保存到 {output_path}")

    # 可視化
    img_with_fit = galaxy_img.copy()
    cv2.ellipse(img_with_fit, ellipse_params, (0, 255, 0), 2)

    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. 原始圖像')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(img_with_fit, cv2.COLOR_BGR2RGB))
    axes[1].set_title('2. 擬合的橢圓投影')
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(deprojected_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('3. 解投影後的正面視圖')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()