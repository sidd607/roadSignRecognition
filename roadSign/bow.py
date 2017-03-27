import cv2
import os
import numpy as np
import pickle
import random

sift = cv2.SIFT()
numClusters = 500
bow = cv2.BOWKMeansTrainer(numClusters)

def get_imlist(paths):
    return [os.path.join(path,f) for path in paths for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.ppm'))]

def readTrafficSigns(rootpath):
	'''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

	Arguments: path to the traffic sign data, for example './GTSRB/Training'
	Returns:   list of images, list of corresponding labels'''
	images = [] # images
	labels = [] # corresponding labels
	# loop over all 42 classes
	for c in range(0,9):
        	prefix = rootpath + '/' + format(c, '02d') + '/' # subdirectory for class
        	print prefix
	        images += get_imlist([prefix])
	        for i in range(len(get_imlist([prefix]))):
	        	labels.append(c)
	return images, labels
    
images,labels =  readTrafficSigns('./FullIJCNN2013_2')
print len(images)
print len(labels)


extractor = cv2.DescriptorExtractor_create("SIFT")
matcher = cv2.DescriptorMatcher_create("FlannBased")
bowimgdesc = cv2.BOWImgDescriptorExtractor(extractor,matcher)
svm_params = dict(kernel_type=cv2.SVM_LINEAR,svm_type=cv2.SVM_C_SVC,C=2.67,gamma=5.383)

knn = cv2.KNearest()
svm = cv2.SVM()
nb = cv2.NormalBayesClassifier()

def getAccuracy(predicted_label, actual_label):
    corrects = [i for i, j in zip(predicted_label, actual_label) if i == j]
    return float(len(corrects))/float(len(actual_label))


for training_img in images:
    img = cv2.imread(training_img)
    #print training_img
    gray = cv2.cvtColor(img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, dsc= sift.detectAndCompute(gray, None)
    bow.add(dsc)


model = bow.cluster()
bowimgdesc.setVocabulary(model)

training_nums = len(images)
featureMatrix = np.empty([training_nums,numClusters],dtype=np.float32)
labels_2 = np.empty([training_nums,1],dtype=np.float32)

print "Time for training"
c = 0
for i in range(len(images)):
	img_path = images[i]
	gray = cv2.imread(img_path,0)
	kp = sift.detect(gray,None)
	feature = bowimgdesc.compute(gray,kp)
	featureMatrix[c] = feature
	labels_2[i] = labels[i]
	c+=1


print 'training'
#knn.train(featureMatrix,labels)
svm.train(featureMatrix,labels_2,params=svm_params)
#nb.train(featureMatrix,labels)
print 'done'

'''print 'Testing'
positives = 0
negatives = 0
testLabels = []
predictedLabels_knn = []
predictedLabels_svm = []
predictedLabels_nb = []
'''
test_img = "FUllIJCNN2013_2/02/00017.ppm"
gray = cv2.imread(test_img,0)
kp = sift.detect(gray,None)
print gray
feature = bowimgdesc.compute(gray,kp)
print feature
#ret, knnResult, neighbours ,dist = knn.find_nearest(feature, 4)
svmResult = svm.predict(feature)

print svmResult

