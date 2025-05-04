import cv2
import numpy as np
import os

def process_image(image_path, debug=False):
    # 讀取原圖
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取 {image_path}")
        return None

    # 複製原圖，用來繪製檢測結果
    result = img.copy()

    # 1. 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 使用中值濾波進行噪聲抑制 (kernel 大小 7)
    denoised = cv2.medianBlur(gray, 7)

    # 3. 使用 CLAHE 進行對比度增強
    #    參數：clipLimit=0.5, tileGridSize=(50, 50)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(50, 50))
    equalized = clahe.apply(denoised)

    # 4. Otsu 閾值法二值化 (反二值化，假設 blemish 較暗)
    ret, thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 根據需求調整閾值 (這邊以 ret - 5 為例)
    adjusted_thresh_val = ret - 5
    ret, thresh = cv2.threshold(equalized, adjusted_thresh_val, 255, cv2.THRESH_BINARY_INV)

    # 5. 形態學處理：先開運算去除小噪點，再閉運算填補斷裂
    kernel_morph = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_morph, iterations=2)

    # 6. 輪廓檢測與篩選
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blemish_detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 根據實際情況調整面積門檻 (此處保留 800 ~ 50000 的區域)
        if area < 800 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        blemish_detected = True
        # 在原圖上畫出紅色邊框
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 在邊框上方標示參數資訊 (面積、寬度、高度)
        info_text = f"Area:{int(area)} W:{w} H:{h}"
        cv2.putText(result, info_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1)

    # 在左上角顯示全域演算法參數 (CLAHE 與 Otsu 閾值)
    global_text = f"CLAHE: clip=0.5, grid=(50,50), Otsu: {ret:.0f}"
    cv2.putText(result, global_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # 顯示檢測結果 (可選 debug 模式)
    if debug:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result

def process_images_in_folder(input_folder, output_folder, debug=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定義要處理的影像副檔名
    valid_ext = [".bmp", ".png", ".jpg", ".jpeg"]
    for filename in os.listdir(input_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_ext:
            continue
        image_path = os.path.join(input_folder, filename)
        result_img = process_image(image_path, debug)
        if result_img is not None:
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_img)
            print(f"處理 {filename} 完成，結果儲存至 {output_path}")

if __name__ == "__main__":
    # 修改為你的輸入與輸出資料夾路徑
    input_folder = r"C:/Users/Terry/Downloads/test-20250423T152606Z-001/test/blemish/"
    output_folder = r"C:/Users/Terry/Downloads/test-20250423T152606Z-001/test/blemish/output_images"
    process_images_in_folder(input_folder, output_folder, debug=True)
