import cv2 
import numpy as np 

img = cv2.imread("./images/rename.jpg")

def applyinvert(image):
    i3 = cv2.bitwise_not(image)
    i1 = cv2.bitwise_and(image, i3)
    i2 = cv2.bitwise_or(image,i1)
    
    i4 = cv2.bitwise_xor(image,i2)
    cv2.imshow("AND", i1)
    cv2.imshow("OR", i2)
    cv2.imshow("NOT", i3)
    cv2.imshow("XOR", i4)
    cv2.waitKey(0)

applyinvert(img)