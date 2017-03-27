import cv2
import os
import numpy as np
import pickle
import random
import signExtraction
import imutils


sift = cv2.SIFT()
numClusters = 700
bow = cv2.BOWKMeansTrainer(numClusters)

knn = cv2.KNearest()
svm = cv2.SVM()
nb 	= cv2.NormalBayesClassifier()

extractor = cv2.DescriptorExtractor_create("SIFT")
matcher = cv2.DescriptorMatcher_create("FlannBased")
bowimgdesc = cv2.BOWImgDescriptorExtractor(extractor,matcher)

def get_imlist(paths):
    return [os.path.join(path,f) for path in paths for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.ppm'))]

def readTrafficSigns(rootpath):
	'''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

	Arguments: path to the traffic sign data, for example './GTSRB/Training'
	Returns:   list of images, list of corresponding labels'''
	images = [] # images
	labels = [] # corresponding labels
	# loop over all 10 classes
	for c in range(0,10):
        	prefix = rootpath + '/' + format(c, '02d') + '/' # subdirectory for class
        	print prefix
	        images += get_imlist([prefix])
	        for i in range(len(get_imlist([prefix]))):
	        	labels.append(c)
	return images, labels

def getAccuracy(predicted_label, actual_label):
    corrects = [i for i, j in zip(predicted_label, actual_label) if i == j]
    return float(len(corrects))/float(len(actual_label))


def train():
	svm_params = dict(kernel_type=cv2.SVM_LINEAR,svm_type=cv2.SVM_C_SVC,C=2.67,gamma=5.383)

	images, labels = readTrafficSigns('./FullIJCNN2013_2')
	print len(images)
	print len(labels)

	for training_img in images:
	    img = cv2.imread(training_img)
	    #print training_img
	    gray = cv2.cvtColor(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	    kp, dsc= sift.detectAndCompute(gray, None)
	    try:
	    	bow.add(dsc)
	    except:
	    	print "couldnt add"
	    	print training_img
	model = bow.cluster()
	bowimgdesc.setVocabulary(model)

	training_num = len(images)
	featureMatrix = np.empty([training_num, numClusters], dtype=np.float32)
	labels_2 = np.empty([training_num, 1], dtype=np.float32)

	print "time for training"

	c=0

	for i in range(len(images)):
		img_path = images[i]
		gray = cv2.imread(img_path, 0)
		kp = sift.detect(gray, None)
		feature = bowimgdesc.compute(gray, kp)
		featureMatrix[c] = feature
		labels_2[i] = labels[i]
		c+=1

	print "training"
	svm.train(featureMatrix, labels_2, params=svm_params)
	print "done!!"


def test(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kp = sift.detect(gray, None)
	feature = bowimgdesc.compute(gray, kp)
	svmResult = svm.predict(feature)
	return svmResult



def predictImage(image):
	imgs, testImage = signExtraction.getContours(image)
	name = '1'
	tmp = []
	for img in imgs:
		cv2.imshow(name, img)
		name += '1'
		#print "Class: ", 
		result = test(img)
		tmp.append(result)
		#print result
	#cv2.imshow("image", image)
	return tmp

if __name__ == "__main__":
	print "TRAFFIC SIGN DETECTION"
	train()

	#testImg = cv2.imread('FUllIJCNN2013_2/02/00017.ppm')
	#print test(testImg)
	images = [	'images/stop.jpg', 
				'images/stop2.jpg', 
				'images/stop3.jpeg', 
				'images/stop4.jpg', 
				'images/stop5.jpg', 
				'images/triangle1.jpg']

	for i in images:
		testImg = cv2.imread(i)
		result = predictImage(testImg)
		print "----------------------------------------"
		print i
		print result
		print "----------------------------------------"

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	


'''
Label for each class:

00 -> 70
01 -> 30
02 -> 50
03 -> 80
04 -> 120
05 -> No Overtake Bus
06 -> NoEntry
07 -> triangle Inverted
08 -> Keep right
09 -> stop
'''