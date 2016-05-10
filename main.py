import cv2
import numpy as np
import imutils

print "Done Importing"

frame = cv2.imread("images/triangle1.jpg")


#cap = cv2.VideoCapture(0)

#while (True):
#	_, frame = cap.read()

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
#cv2.imshow('dst', dst)

#--------------------------------------------------------
 
ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)

#cv2.imshow('thresh', thresh)



#gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
#gray = dst

#ret, thresh = cv2.threshold(gray, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow('image', image)


for cnt in contours:
	x, y, w, h = cv2.boundingRect(cnt)
	aspect_ratio = float(w)/h
	#print aspect_ratio
	if aspect_ratio < 1.2 and aspect_ratio > 0.7 and cv2.contourArea(cnt) > 700:
		print cv2.contourArea(cnt)
		#cv2.drawContours(image, [cnt], -1, (0,255,0), 3)
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		center = (int(x),int(y))
		radius = int(radius)
		cv2.circle(image,center,radius,(0,255,0),2)
		print center
		#print aspect_ratio

cv2.imshow('contours', image)


#--------------------------------------------------------




#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
#cv2.imshow('hsv', hsv)



#if cv2.waitKey(1) & 0xFF == ord('q'):
#	break
cv2.waitKey(0)

cv2.destroyAllWindows()