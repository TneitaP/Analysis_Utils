import cv2 
import numpy as np



if __name__ == "__main__":
    template_img = cv2.imread("demo_images/template.png")
    template_img = cv2.resize(template_img, (640, 640))

    # template_img = cv2.medianBlur(template_img,5)

    cv2.imshow("MedianBlur", template_img)
    cv2.waitKey(-1)
