import numpy
import math
import os
import gc
import cv2
import pandas
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle
import random

class CNNSequence(Sequence):
    def __init__(self,datacsv,indexes,batchSize,labelName,shuffle=True,augmentations = False):
        # author Rebecca Hisey
        self.inputs = numpy.array([os.path.join(datacsv["Folder"][x],datacsv["FileName"][x]) for x in indexes])
        self.batchSize = batchSize
        self.labelName = labelName
        self.labels = numpy.array(sorted(datacsv[self.labelName].unique()))
        self.targets = numpy.array([self.convertTextToNumericLabels(datacsv[labelName][x]) for x in indexes])
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            shuffledInputs,shuffledTargets = shuffle(self.inputs,self.targets)
            self.inputs = shuffledInputs
            self.targets = shuffledTargets


    def __len__(self):
        # author Rebecca Hisey
        length = len(self.inputs) / self.batchSize
        length = math.ceil(length)
        return length

    def convertTextToNumericLabels(self, textLabel):
        label = numpy.zeros(len(self.labels))
        labelIndex = numpy.where(self.labels == textLabel)
        label[labelIndex] = 1
        return label

    def rotateImage(self,image,angle = -1):
        if angle < 0:
            angle = random.randint(1, 359)
        center = tuple(numpy.array(image.shape[1::-1])/2)
        rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
        rotImage = cv2.warpAffine(image,rot_mat,image.shape[1::-1],flags=cv2.INTER_LINEAR)
        return rotImage

    def flipImage(self,image,axis):
        return cv2.flip(image, axis)

    def readImage(self,file):
        image = cv2.imread(file)
        try:
            resized_image = cv2.resize(image, (224, 224))
            normImg = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            preprocessingMethod = random.randint(0, 3)
            del image
            del resized_image
            if self.augmentations and preprocessingMethod == 0:
                # flip along y axis
                return cv2.flip(normImg, 1)
            elif self.augmentations and preprocessingMethod == 1:
                # flip along x axis
                return cv2.flip(normImg, 0)
            elif self.augmentations and preprocessingMethod == 2:
                # rotate
                angle = random.randint(1, 359)
                rotImage = self.rotateImage(normImg, angle)
                return rotImage
            else:
                return normImg
        except:
            print(file)

    def __getitem__(self,index):
        # author Rebecca Hisey
        startIndex = index*self.batchSize
        indexOfNextBatch = (index + 1)*self.batchSize
        inputBatch = [self.readImage(x) for x in self.inputs[startIndex : indexOfNextBatch]]
        outputBatch = [x for x in self.targets[startIndex : indexOfNextBatch]]
        inputBatch = numpy.array(inputBatch)
        outputBatch = numpy.array(outputBatch)
        return (inputBatch,outputBatch)
