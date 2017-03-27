import imutils
import cv2
import numpy as np

def test():
	print "Hello World"

def within(cnt, contours):
	print len(contours)

	(x1, y1), radius1 = cv2.minEnclosingCircle(cnt)
	for c in contours:
		(x2, y2), radius2 = cv2.minEnclosingCircle(c)
		distance = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1)
		radDiff = (radius1 - radius2)*(radius1 - radius2)
		if radDiff > distance and radius1 < radius2:
			return True
	return False

def getContours(image):
	frame = image
	frame = imutils.resize(frame, width=600)

	image = frame

	blurred = cv2.GaussianBlur(frame, (11, 11), 0)

	#cv2.imshow('blurred', blurred)

	frame = blurred

	lowerRed1 = np.array([0, 100, 100])
	upperRed1 = np.array([20, 255, 255])
	lowerRed2 = np.array([159, 100, 100])
	upperRed2 = np.array([179, 255, 255])

	lowerBlue = np.array([100, 100, 100])
	upperBlue = np.array([120, 255, 255])

	thresholdContour = np.array([128, 128, 128])

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	mask1 = cv2.inRange(hsv, lowerBlue, upperBlue)
	mask2 = cv2.inRange(hsv, lowerRed1, upperRed1)
	mask3 = cv2.inRange(hsv, lowerRed2, upperRed2)

	res = cv2.bitwise_and(frame, frame, mask = mask1)
	dst = cv2.addWeighted(mask1, 1, mask2, 1, 0)
	dst = cv2.addWeighted(dst, 1, mask3, 1, 0)

	ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)

	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	imgs = []
	c = '1'
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		aspect_ratio = float(w)/h

		if aspect_ratio < 1.2 and aspect_ratio > 0.7 and cv2.contourArea(cnt) > 700:
			print cv2.contourArea(cnt)
			#if within(cnt, contours) == False:
			(x,y), radius = cv2.minEnclosingCircle(cnt)
			center = (int(x), int(y))
			radius = int(radius)

			if center[0] - radius < 0:
				x = 0
			else:
				x = center[0] - radius

			if center[1] - radius < 0:
				y = 0
			else:
				y = center[1] - radius
			w = 2*radius
			h = 2*radius

			tmp = image[y:y+h, x:x+h]
			imgs.append(tmp)
			#cv2.circle(image, center, radius, (255, 0, 0), 2)	  

	return imgs, image