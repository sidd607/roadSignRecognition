import cv2
import numpy as np
import imutils

print "Done Importing"

frame = cv2.imread("images/stop3.jpeg")
frame = imutils.resize(frame, width=600)

image = frame

blurred = cv2.GaussianBlur(frame, (11,11), 0)
cv2.imshow('blurred', blurred)

frame = blurred

lowerRed1 = np.array([0, 100, 100])
upperRed1 = np.array([20, 255, 255])

thresholdContour  = np.array([128, 128, 128])

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower_blue = np.array([159,135,135])
upper_blue = np.array([179,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask2 = cv2.inRange(hsv, lowerRed1, upperRed1)
cv2.imshow('mask2', mask2)
res = cv2.bitwise_and(frame,frame, mask= mask)

dst = cv2.addWeighted(mask,1,mask2,1,0)
#dst = cv2.inRange(dst,  , mask=dst)
cv2.imshow('dst', dst)

#--------------------------------------------------------
 
ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('thresh', thresh)



#gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#gray = dst

#ret, thresh = cv2.threshold(gray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('image', image)
cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow('contours', image)


#--------------------------------------------------------




cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.imshow('hsv', hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()