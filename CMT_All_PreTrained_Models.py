# Link to the used database
# https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-test-set/code
# the folder of database should look like:
# CMT Dataset folder that has two other folders
# benign and malignant
# benign folder has images of benign class
# malignant folder has images of malignant class
# so we have two classes benign and malignant
""" ******************************************************************************** """
import time
start = time.time()
print("The starttime is :", (start) * 10**3/60, " Seconds")
""" ******************************************************************************** """
import cv2
import pandas as pd
import tensorflow as tf
from   tensorflow.keras import layers
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
import graphviz
import tensorflow_datasets as tfds
import matplotlib.image as mpimg

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random                  import shuffle
from sklearn.model_selection import train_test_split 
from tensorflow.keras.models import Sequential, save_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers          import Adam

from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3
#from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import DenseNet201, Xception, EfficientNetB0

from skimage.transform                    import resize
from tensorflow.keras.metrics             import Precision, Recall
from PIL                                  import Image as im 
from tensorflow.keras.utils               import load_img
from tensorflow.keras.utils import to_categorical
print("tensorflow version is: ", tf.__version__)

""" ******************************************************************************** """
class Configurations:

  IMAGE_WIDTH  = 128
  IMAGE_HEIGHT = 128
  BATCH_SIZE   = 64  
  test_size    = 0.20
  learning_rate= 0.001
  batch_size   = 16
  validation_perentage = 0.05 #5%

  #All these parameter should be changed based on your dataset
  Data_Path    = "E:\\Research_Abo\\Datasets\\animals-13-01563-s001\\Python Code\\CMT Dataset\\"
  CMT_labels   = ['benign', 'malignant']
  CMT_values   = [0       ,     1      ]
  Num_Classes  = len(CMT_labels)
  CNN_Model    = ['Basic', 'VGG', 'ResNet50','MobileNet',  'Inception','InceptionResNet', 'DenseNet201', 'Xception', 'EfficientNet']
  Current_CNN  = 'EfficientNet'
  epochs       = MobileNet
  
  model_path   = Current_CNN + '_' + str(epochs) + 'Epochs'
""" ******************************************************************************** """
class Data:

  def Load_Data_Labels(self, Data_Path, IMAGE_WIDTH, IMAGE_HEIGHT, ClassLabel1):

    # Download the training and validation data
    filenames  = os.listdir(Data_Path) #Contains two folder one for images of each class
    print("Folder Names = Class Labels: : " , filenames)

    Labels     = []
    data       = []

    for filename in filenames:

      images = os.listdir(Data_Path + filename)
      
      for image in images:
        image_path = Data_Path + "\\"+ filename +"\\"+image

        #Create Labels:
        Label  = filename
        if Label == ClassLabel1:
          Label = 0 #Put 0  for benign
        else:
          Label = 1 #Put 1  for malignant

        #Create Image  
        image = cv2.imread(image_path)

        #Resize image into 128*128 
        image = cv2.resize(image, (IMAGE_WIDTH , IMAGE_HEIGHT),interpolation = cv2.INTER_AREA)
        #image = resize(image, (IMAGE_WIDTH , IMAGE_HEIGHT,1))

        #image = cv2. cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = np.expand_dims(image, 2)

        #Normalize image pixel values between 0 and 255.
        image = image.astype('float') / 255.0
        
        #Append both image, and labels in data array
        data.append([image, Label])

    print("Number of samples from both classes are: len (data)" , len(data))

    return data

  def split_data(self, data, test_size, IMAGE_WIDTH, IMAGE_HEIGHT):

    X = np.array([i[0] for i in data]).reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 3)
    y = np.array([i[1] for i in data])
    
    x_train, x_test, y_train, y_test = train_test_split(X,y , random_state= 104, test_size =test_size,shuffle     = True)

    print ("y_train shape: " , y_train.shape) #(844,)
    print ("y_test shape : " , y_test.shape)  #(212,)
    print ("x_train shape: " , x_train.shape) #(844, 128, 128, 3)
    print ("x_test shape : " , x_test.shape)  #(212, 128, 128, 3)    

    return x_train, y_train, x_test, y_test

  def plot_label_count(self, data, CMT_labels):

    #Plot count of label of each class in the dataset
    labels         = np.array([i[1] for i in data])
    elements_count = {}

    for element in labels:
      if element in elements_count:
          elements_count[element] += 1
      else:
          elements_count[element] = 1

    for key, value in elements_count.items():
      print(f"{key}: {value}")
     
    x = [CMT_labels[0]    , CMT_labels[1]]
    y = [elements_count[0], elements_count[1]]

    for i in range(len(x)):
        plt.text(i, y[i]//2,y[i], ha = 'center')

    plt.bar(x, y, color = 'red')
    plt.title("The count of Each class in CMT dataset")
    plt.xlabel("CMT Classes")
    plt.ylabel("Count of Each Class")
    plt.show()

  def plot_data_samples3(self, x_train, y_train , label_name, label_value, sizeofplot):

      fig, ax  = plt.subplots(sizeofplot,sizeofplot, figsize=(5,5)) 
      result   = np.where(y_train[0:30] == label_value)[0]
      index    = 0
      for i in range(sizeofplot):
        for j in range (sizeofplot):
            ax[i][j].imshow(tf.squeeze(x_train[result[index]]), cmap=plt.cm.gray) 
            ax[i][j].set_title("Class : " + label_name) 
            plt.axis("off")
            index =  index+1                                            
      fig.tight_layout()
      plt.show()
""" ******************************************************************************** """
class CNN:

  def One_Hot_Encoding(self, y_train, num_classes ):
    
    y_train = to_categorical(y_train , num_classes=num_classes)

    """
    print("y_train[0] after one-hot encoding : " , y_train[0]) #[1 0]
    print("y_train shape as one-hot encoding : " , y_train[0].shape)
    """
    return y_train
  
  def model_save(self, model , model_path):
    
    model.save(model_path)

  def plot_history_train(self, history):#call after model.fit

      plt.figure(figsize=(12, 5))

      # Plot loss
      plt.subplot(1, 2, 1)
      plt.plot(history.history['loss']    , label='train_loss')
      plt.plot(history.history['val_loss'], label='val_loss')
      plt.legend()
      plt.xlabel('Epochs')
      plt.ylabel('Loss')
      plt.title('Loss over Epochs')

      # Plot accuracy
      plt.subplot(1, 2, 2)
      plt.plot(history.history['accuracy']    , label='train_accuracy')
      plt.plot(history.history['val_accuracy'], label='val_accuracy')
      plt.legend()
      plt.xlabel('Epochs')
      plt.ylabel('Accuracy')
      plt.title('Accuracy over Epochs')
      plt.legend(loc='lower right')

      plt.tight_layout()
      plt.show()

  def plot_confusion_matrix(self, model, x_test, y_test):
      
      test_predictions       = model.predict(x_test)
      test_predicted_labels  = np.argmax(test_predictions, axis=1)
      test_true_labels       = np.argmax(y_test, axis=1)
      cm                     = confusion_matrix(test_true_labels, test_predicted_labels)
      cmd                    = ConfusionMatrixDisplay(confusion_matrix=cm)
      cmd.plot(include_values= True, cmap='viridis', ax=None, xticks_rotation='horizontal')
      plt.show()
     
  def plot_classification_report(self, model, x_test, y_test,CMT_labels, Num_Classes):
      
      predictions       = model.predict(x_test)
      predicted_classes = [np.argmax(pred) for pred in predictions]
      
      classes           = CMT_labels
      predicted_classes = self.One_Hot_Encoding(predicted_classes, Num_Classes)

      print(classification_report(y_test, predicted_classes))

      report            = classification_report(y_test, predicted_classes, target_names=classes, output_dict=True)
      #report            = classification_report(y_test, predicted_classes)#, target_names=classes, output_dict=True)
      metrics           = {label: report[label]        for label in classes if label in report}
      precision         = [metrics[label]['precision'] for label in classes]
      recall            = [metrics[label]['recall']    for label in classes]
      f1_score          = [metrics[label]['f1-score']  for label in classes]

      data = {
              'Precision': precision,
              'Recall'   : recall,
              'F1-Score' : f1_score
              }

      df   = pd.DataFrame(data, index=classes)

      plt.figure(figsize=(10, 6))
      sns.heatmap(df, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
      plt.title('Classification Report')
      plt.xlabel('Metrics')
      plt.ylabel('Classes')
      plt.show()
  
  def plot_model (self, model):
      """It has a problem while running .."""
      tf.keras.utils.plot_model(
        model,
        to_file         = "model.png",
        show_shapes     = True,
        show_layer_names= True,
        rankdir         = "TB",
        expand_nested   = True,
        dpi             = 96,
        )
  
  def plot_predictions(self, model, x_test, y_test):

    predictions    = model.predict(x_test)
    sample_classes = np.argmax(predictions, axis = 1)
    sizeofplot     = 3 #subplot 3*3
    fig, ax        = plt.subplots(sizeofplot,sizeofplot, figsize=(5,5)) 
    index          = 0  

    for i in range(sizeofplot):
        for j in range(sizeofplot):

          if y_test[index].argmax() == 0:#argmax since one hot
              true = "benign"
          else:
              true = "malignant"
          
          if sample_classes[index] == 0:
              pred = "benign"
          else:
              pred = "malignant"

          color   = "green"
          if true != pred:
             color="red"

          ax[i][j].imshow(tf.squeeze(x_test[index]), cmap=plt.cm.gray) 
          ax[i][j].set_title("True: " + true + " Predicted: "+ pred, color = color ) 
          plt.axis("off")
          index = index + 1

    fig.tight_layout()
    plt.show()
    cv2.waitKey(50) 
""" ******************************************************************************** """
class Models:
  
  def CNN_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT, Num_classes):
    print ("******************************************** Basic CNN Model *********************************************")
    model = tf.keras.models.Sequential()

    # Input 0 of layer "conv2d" is incompatible with the layer: expected axis -1 of input shape to have value 1, 
    # but received input with shape (None, 128, 128, 3)        
    model.add(layers.Conv2D(8, (3, 3), padding='same',activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(16, (3, 3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    return model
    
  def VGG_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
      print ("******************************************** VGG Model *********************************************")
      vgg       = VGG16(input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3) , weights = 'imagenet', include_top = False) 
      return vgg

  def ResNet_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
      print ("******************************************** ResNet Model *********************************************")
      resnet    =  ResNet50(input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,3), weights = 'imagenet',include_top=False)   
      return resnet

  def MobileNet(self, IMAGE_WIDTH, IMAGE_HEIGHT):
      MobileNet = MobileNetV2      (input_shape = (IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights = 'imagenet', include_top = False)
      return MobileNet

  def Inception_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
      print ("******************************************** Inception Model *********************************************")
      inception = InceptionV3(input_shape = (IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights = 'imagenet', include_top = False )
      return inception
  
  def InceptionResNet_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
      print ("******************************************** InceptionResNet Model *********************************************")
      inceptionResnet = InceptionResNetV2(input_shape = (IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights = 'imagenet', include_top = False )
      return inceptionResnet
  
  def DenseNet_Model(self, IMAGE_WIDTH , IMAGE_HEIGHT ):
    print ("******************************************** DenseNet Model *********************************************")
    DenseNet = DenseNet201(input_shape=(IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights='imagenet', include_top=False, pooling='avg')
    return DenseNet

  def Exception_model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
    print ("******************************************** Exception Model *********************************************")
    Exception = Xception(input_shape = (IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights = 'imagenet', include_top = False )
    return Exception

  def EfficientNet_Model(self, IMAGE_WIDTH, IMAGE_HEIGHT):
    print ("******************************************** EfficientNet Model *********************************************")
    EfficientNet = EfficientNetB0(input_shape = (IMAGE_WIDTH , IMAGE_HEIGHT ,3), weights = 'imagenet', include_top = False )
    return EfficientNet
  
  def choose_Model(self, Current_CNN, CNN_Model, IMAGE_WIDTH, IMAGE_HEIGHT, Num_Classes):
     
    if Current_CNN == 'Basic':
        model = self.CNN_Model(IMAGE_WIDTH, IMAGE_HEIGHT, Num_Classes)

    elif Current_CNN == 'VGG':
        model = self.VGG_Model(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    elif Current_CNN == 'ResNet50':
        model = self.ResNet_Model(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    elif Current_CNN == 'MobileNet':
        model = self.MobileNet(IMAGE_WIDTH, IMAGE_HEIGHT)

    elif Current_CNN == 'Inception':
        model = self.Inception_Model(IMAGE_WIDTH, IMAGE_HEIGHT)

    elif Current_CNN == 'InceptionResNet':
        model = self.InceptionResNet_Model(IMAGE_WIDTH, IMAGE_HEIGHT)

    elif Current_CNN == 'DenseNet201':
        model = self.DenseNet_Model(IMAGE_WIDTH, IMAGE_HEIGHT)

    elif Current_CNN == 'Xception':
        model = self.Exception_model(IMAGE_WIDTH, IMAGE_HEIGHT)

    elif Current_CNN == 'EfficientNet':
      model = self.EfficientNet_Model(IMAGE_WIDTH, IMAGE_HEIGHT)
    
    return model

  def create_model(self, model, num_classes):
     
      for layer in model.layers:
            layer.trainable = False

      x     = Flatten()(model.output)
      x     = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.
      model = Model(inputs = model.input, outputs = x)

      model.compile(
                      optimizer= 'Adam',
                      loss     = 'categorical_crossentropy',
                      metrics  = 'accuracy'
                    )

      #model.summary()

      return model
""" ******************************************************************************** """
if __name__== '__main__':

    Config  = Configurations()
    Data    = Data()
    CNN     = CNN()
    Models  = Models()

    ##Save images, and labels in array called data
    data                                      = Data.Load_Data_Labels(Config.Data_Path,Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.CMT_labels[0])

    ##Split data into train and test with test_size ratio
    x_train, y_train, x_test, y_test_original = Data.split_data(data, Config.test_size, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT)

    ##Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_train       = CNN.One_Hot_Encoding(y_train, Config.Num_Classes)
    y_test        = CNN.One_Hot_Encoding(y_test_original, Config.Num_Classes)

    ##Create the CNN model
    model         = Models.choose_Model(Config.Current_CNN, Config.CNN_Model, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT, Config.Num_Classes)
    model         = Models.create_model(model, Config.Num_Classes)

    ##Train the CNN model on train data and labels: Reserve 5% samples of train data for validation
    nsamples_val  = int(Config.validation_perentage*len(x_train))
    x_train        = x_train[:-nsamples_val]
    y_train        = y_train[:-nsamples_val]
    x_val          = x_train[-nsamples_val:]#last 100 samples in x_train 
    y_val          = y_train[-nsamples_val:]
    history        = model.fit(x=x_train ,y=y_train ,epochs=Config.epochs, batch_size=Config.batch_size, validation_data=(x_val, y_val))
    print ("nsamples_val     : " , nsamples_val)
    print ("nsamples_train   : " , len(x_train) - nsamples_val)

    ##Draw the results of accuray, and loss
    CNN.plot_history_train(history)

    ##Evaluate the model on the test data using `evaluate`
    print ("Evaluating the model for the testing data ... ")
    results = model.evaluate(x_test, y_test, batch_size=Config.batch_size)
    print("test loss, test acc:", results)

    ##Plot the confusion matrix of this model for the test data
    CNN.plot_confusion_matrix( model, x_test, y_test)
    
    ##Plot the classification report of this model for the test data
    CNN.plot_classification_report(model, x_test, y_test, Config.CMT_labels, Config.Num_Classes)

    ##Show the predition results as images
    CNN.plot_predictions(model , x_test, y_test)

    ##Plot count of label of each class in the dataset
    Data.plot_label_count(data , Config.CMT_labels)
    
    ##Show some samples of the data
    Data.plot_data_samples3(x_train, y_train , Config.CMT_labels[0], Config.CMT_values[0], 2)
    Data.plot_data_samples3(x_train, y_train , Config.CMT_labels[1], Config.CMT_values[1], 2)

    end = time.time()
    print("The duration of excuting this projct is :", (end-start) /60, " Minutes")
    
    """ ******************************************************************************** """
