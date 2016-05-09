import cv2
import numpy as np
import imutils

frame = cv2.imread("stop5.jpg")

frame = imutils.resize(frame, width=600)
blurred = cv2.GaussianBlur(frame, (11,11), 0)
cv2.imshow('blurred', blurred)

frame = blurred

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([159,135,135])
upper_blue = np.array([179,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(frame,frame, mask= mask)
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.imshow('hsv', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()