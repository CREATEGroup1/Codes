import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import InceptionV3,ResNet50, Xception, ResNet101, EfficientNetB4, DenseNet121, ResNet50V2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Activation
from tensorflow.keras.models import *

from vit_keras import vit
import tensorflow_addons as tfa

#tensorflow.compat.v1.enable_eager_execution()

class CNN():
    def __init__(self):
        self.cnnModel = None
        self.cnnLabels = None

    def loadModel(self,modelFolder):
        self.cnnModel = self.loadCNNModel(modelFolder)

        with open(os.path.join(modelFolder,"cnn_labels.txt"),"r") as f:
            self.cnnLabels = f.readlines()
        self.cnnLabels = numpy.array([x.replace("\n","") for x in self.cnnLabels])

    def loadCNNModel(self,modelFolder):
        structureFileName = 'resnet50.json'
        weightsFileName = 'resnet50.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        return model

    def predict(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224)) #MobileNet
        #resized = cv2.resize(image, (299, 299))  #InceptionV3
        normImage = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normImage = numpy.expand_dims(normImage, axis=0)

        toolClassification = self.cnnModel.predict(numpy.array(normImage))
        labelIndex = numpy.argmax(toolClassification)
        label = self.cnnLabels[labelIndex]
        networkOutput = str(label) + str(toolClassification)
        return networkOutput

    def createCNNModel(self,imageSize,num_classes):
        model = tensorflow.keras.models.Sequential()
        model.add(ResNet50(weights='imagenet',include_top=False,input_shape=imageSize,pooling='avg'))
        #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))
        return model

    def createViTModel(self,imageSize,num_classes):
        vit_model = vit.vit_b32(
            image_size=224,
            activation='softmax',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            classes=num_classes)

        model = tf.keras.Sequential([
            vit_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes, 'softmax')
        ],
            name='vision_transformer')
        return model

    # def createSSLModel(self, num_classes, pretrained_path):
    #     _, base_model = self.get_resnet_simclr(256,128,50, pretrained_path)
    #     base_model.trainable = True
    #     model = tensorflow.keras.models.Sequential()
    #     model.add(base_model)
    #     model.add(GlobalAveragePooling2D())
    #     #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
    #     model.add(layers.Dense(512,activation='relu'))
    #     model.add(layers.Dense(num_classes,activation='softmax'))
    #     return model

    # def createSSLModel(self, num_classes, pretrained_path):
    #
    #     backbone = tensorflow.keras.Model(
    #         simsiam.encoder.input, simsiam.encoder.get_layer("backbone_pool").output
    #     )
    #     .trainable = True
    #     model = tensorflow.keras.models.Sequential()
    #     model.add(base_model)
    #     model.add(GlobalAveragePooling2D())
    #     #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
    #     model.add(layers.Dense(512,activation='relu'))
    #     model.add(layers.Dense(num_classes,activation='softmax'))
    #
    #     return model

    # Architecture utils
    def get_resnet_simclr(self, hidden_1, hidden_2, hidden_3, pretrained_path):
        base_model = ResNet101(include_top=False, weights=None, input_shape=(224, 224, 3))
        base_model.trainable = True
        inputs = Input((224, 224, 3))
        h = base_model(inputs, training=True)
        h = GlobalAveragePooling2D()(h)

        projection_1 = Dense(hidden_1)(h)
        projection_1 = Activation("relu")(projection_1)
        projection_2 = Dense(hidden_2)(projection_1)
        projection_2 = Activation("relu")(projection_2)
        projection_3 = Dense(hidden_3)(projection_2)

        resnet_simclr = Model(inputs, projection_3)

        resnet_simclr.load_weights(pretrained_path)
        return resnet_simclr, base_model

    def saveModel(self,trainedCNNModel,saveLocation):
        JSONmodel = trainedCNNModel.to_json()
        structureFileName = 'resnet50.json'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)
