import numpy as np
import cv2
from glob import glob

def color_detector(filename):
        # 1) 이미지 읽기
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {filename}")

    # 2) 흑색 테두리(모니터) 검출
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'G'
    # 가장 큰 contour 선택
    lc = max(contours, key=cv2.contourArea)

    # 3) 마스크 생성 (컨투어 내부만 True)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [lc], -1, color=255, thickness=cv2.FILLED)
    inside = mask.astype(bool)

    # 4) HSV 변환 및 채널 분리
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    # 5) 조건에 맞는 픽셀 마스크
    valid = inside & (s > 50) & (v > 50)
    h_valid = h[valid]

    # 6) R/G/B 카운트 (vectorized)
    R_count = np.count_nonzero((h_valid > 160) | (h_valid < 20))
    G_count = np.count_nonzero((h_valid > 40) & (h_valid < 80))
    B_count = np.count_nonzero((h_valid > 100) & (h_valid < 140))

    # 7) 가장 큰 값의 키 반환
    counts = {'R': R_count, 'G': G_count, 'B': B_count}
    return max(counts, key=counts.get)

    
if __name__ == '__main__':
    result=[]
    for filename in sorted(glob('public_imgs/*.PNG')): 
        result.append(color_detector(filename))
    print(result)