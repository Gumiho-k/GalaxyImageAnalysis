import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_galaxy_properties(image_path):
    """
    從圖像中讀取星系，並找到其輪廓和中心點。
    這一步是為了將星系與黑色背景分離，並獲取變換所需的中心坐標。

    Args:
        image_path (str): 輸入圖像的路徑。

    Returns:
        tuple: (僅包含星系的圖像, 橢圓中心點, 原始圖像尺寸) or (None, None, None) if failed.
    """
    # 讀取圖像
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖像 {image_path}")
        return None, None, None

    h, w = img.shape[:2]

    # 轉換為灰度圖並進行二值化以找到輪廓
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) # 使用較低的閾值以確保捕捉到暗淡的邊緣

    # 查找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("在圖像中未找到輪廓。")
        return None, None, None

    # 假設最大的輪廓是我們的目標星系
    largest_contour = max(contours, key=cv2.contourArea)

    if len(largest_contour) < 5:
        print("輪廓點太少，無法擬合橢圓。")
        return None, None, None

    # 擬合橢圓以獲得精確的中心點
    # 我們只使用這個橢圓的中心點，角度和軸長將由用戶提供
    ellipse_params = cv2.fitEllipse(largest_contour)
    center = ellipse_params[0]

    # 創建一個掩碼，只保留星系區域的像素
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    galaxy_only = cv2.bitwise_and(img, img, mask=mask)

    return galaxy_only, center, (h, w)


def restore_galaxy_view(image, center, inclination_deg, pa_deg, output_size):
    """
    根據用戶輸入的傾角和位置角，對圖像進行解投影（還原）變換。

    Args:
        image (numpy.ndarray): 僅包含星系的圖像。
        center (tuple): 星系的中心坐標 (x, y)。
        inclination_deg (float): 用戶輸入的傾角。
        pa_deg (float): 用戶輸入的位置角。
        output_size (tuple): 輸出圖像的尺寸 (height, width)。

    Returns:
        numpy.ndarray: 還原後的正面視圖圖像。
    """
    # 根據用戶輸入的傾角計算拉伸因子
    # 這是解投影的核心數學步驟
    # stretch_factor = 1 / cos(inclination)
    # 處理90度的特殊情況，避免除以零
    if inclination_deg >= 90.0:
        inclination_deg = 89.9 # 防止無限拉伸
    
    stretch_factor = 1.0 / np.cos(np.deg2rad(inclination_deg))

    h, w = output_size
    
    # 構建仿射變換矩陣 M
    # 變換順序：
    # 1. 將圖像旋轉-pa_deg，使星系的長軸與水平軸對齊。
    # 2. 沿垂直方向（短軸方向）拉伸 stretch_factor 倍。
    # 3. 將圖像旋轉回 +pa_deg。
    # 4. 整個過程圍繞圖像中心點進行。

    # OpenCV 的 getRotationMatrix2D 和 warpAffine 可以組合完成這個任務
    
    # 步驟1: 創建一個從原點出發的完整變換矩陣
    # 先旋轉對齊
    M = cv2.getRotationMatrix2D((0,0), -pa_deg, 1)
    # 再拉伸
    M[1,:] = M[1,:] * stretch_factor
    # 再旋轉回去
    M_rot_back = cv2.getRotationMatrix2D((0,0), pa_deg, 1)
    M_combined = M_rot_back @ np.vstack([M, [0,0,1]])
    
    # 步驟2: 調整變換矩陣，使其圍繞圖像中心點 `center` 旋轉，而不是原點(0,0)
    M_combined[0, 2] += center[0] - (M_combined[0, 0] * center[0] + M_combined[0, 1] * center[1])
    M_combined[1, 2] += center[1] - (M_combined[1, 0] * center[0] + M_combined[1, 1] * center[1])

    # 步驟3: 應用最終的仿射變換
    restored_image = cv2.warpAffine(image, M_combined, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

    return restored_image


def main():
    """
    主執行函數：獲取用戶輸入並執行圖像還原。
    """
    image_path = 'separated_galaxies/NGC_3627.jpg'

    # 步驟 1: 分析原始圖像，獲取星系主體和中心點
    galaxy_img, center_point, img_size = find_galaxy_properties(image_path)
    if galaxy_img is None:
        return

    # 步驟 2: 獲取用戶輸入的角度
    try:
        print("--- 請輸入您要用於還原圖像的歐拉角 ---")
        user_inclination = float(input("請輸入傾角 (Inclination) [0-90 度]: "))
        user_pa = float(input("請輸入位置角 (Position Angle) [0-180 度]: "))
        
        # 驗證輸入範圍
        if not (0 <= user_inclination <= 90):
            print("錯誤：傾角必須在 0 到 90 度之間。")
            return
        if not (0 <= user_pa <= 180):
            print("錯誤：位置角必須在 0 到 180 度之間。")
            return

    except ValueError:
        print("無效輸入，請確保您輸入的是數字。")
        return

    # 步驟 3: 根據用戶輸入的角度執行圖像還原
    restored_img = restore_galaxy_view(galaxy_img, center_point, user_inclination, user_pa, img_size)

    # 步驟 4: 保存並顯示結果
    output_path = f'restored_i{int(user_inclination)}_pa{int(user_pa)}.jpg'
    cv2.imwrite(output_path, restored_img)
    print(f"\n操作成功！還原後的圖像已保存為 '{output_path}'")

    # 可視化
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 顯示原始圖像
    original_img_display = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    axes[0].imshow(original_img_display)
    axes[0].set_title('1. 原始觀測圖像 (Original Photo)')
    axes[0].axis('off')

    # 顯示還原後的圖像
    restored_img_display = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    axes[1].imshow(restored_img_display)
    axes[1].set_title(f'2. 還原後的正面視圖 (Restored View)\ni={user_inclination}°, PA={user_pa}°')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()






