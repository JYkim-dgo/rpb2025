import cv2
import numpy as np

# --- 1. 이미지 읽기 -----------------------------------
img = cv2.imread('public_22.PNG')          # BGR 이미지
orig = img.copy()

# --- 2. 검정색 테두리 검출을 위한 전처리 -------------
# 2-1) 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2-2) 임계값: 어두운(거의 검정) 픽셀만 흰색으로 (반전)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# 2-3) 노이즈 제거를 위한 모폴로지 클로징
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# --- 3. 외곽 윤곽선(컨투어) 검출 -----------------------
contours, _ = cv2.findContours(clean,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)

largest_contour = max(contours, key=cv2.contourArea)
#-----------------------------------------------------------
mask = np.zeros_like(gray, dtype=np.uint8)          
cv2.drawContours(mask, [largest_contour], 
                 contourIdx=-1,   # 리스트에 contour 하나만 있으므로
                 color=255,        # 내부를 255로 채움
                 thickness=cv2.FILLED)

# 2) 내부 픽셀의 (y, x) 좌표 추출
ys, xs = np.where(mask == 255)   # ys: 행 인덱스 리스트, xs: 열 인덱스 리스트
coords = list(zip(xs, ys))       # (x, y) 튜플 리스트

# (선택) HSV 값으로 보고 싶다면
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_values = hsv[ys, xs]   

R = 0
G = 0
B = 0
for i in hsv_values:
    if i[1] >50 and i[2] > 50:
        if i[0] > 160 or i[0] < 20:
            R += 1
        elif 40 < i[0] < 80:
            G +=1
        elif 100 < i[0] < 140:
            B +=1

vals = {'R': R, 'G': G, 'B': B}
answer = max(vals, key=vals.get)
print(answer)

    


