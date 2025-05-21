# !/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np
import cv2
from glob import glob


def color_detector(img):
        # 1) 이미지 읽기
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
    B_count = np.count_nonzero((h_valid > 100) & (h_valid < 140))

    total_valid = h_valid.size
    others_count = total_valid - R_count - B_count
    # 7) 가장 큰 값의 키 반환
    counts = {'+1': R_count, '0': others_count, '-1': B_count}
    return max(counts, key=counts.get)

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # listen image topic
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
    
            # determine background color
            # TODO 
            msg.frame_id=color_detector(img)
            
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)
            # determine the color and assing +1, 0, or, -1 for frame_id
            # msg.frame_id = '+1' # CCW 
            # msg.frame_id = '0'  # STOP
            # msg.frame_id = '-1' # CW 
   

if __name__ == "__main__":
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


