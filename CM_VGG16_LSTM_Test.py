# common library
import numpy as np
import os
import time
import cv2
import matplotlib.pyplot as plt

# tensorflow library
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, GRU, LSTM, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation, Dropout, SimpleRNN
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, MaxPooling2D
# from keras.layers import TimeDistributed, GRU, Dense, Dropout
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

from keras.applications.vgg16 import VGG16

# reading data
import csv
import threading

# Label encoder
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import os
os.chdir('C:\\Users\\ASUS\\Desktop\\FYP\\VGG16_RNN_LSTM_SignLanguageRecognition-main')
print(os.getcwd())

# some global params
SIZE = (64, 64)

# SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 10
BS = 20

NUM_THREADS = 8

#preprocessing
LENGTH_TRIM = 130
classes_file = 'classes.npy'

# resize image by dsize = (128, 128)
def resizeImage(img, dsize):
  sizeImg = (img.shape[0],img.shape[1])
  if sizeImg != dsize and dsize != (None, None):
    img = cv2.resize(img,dsize)
  return img

# remove left right background
def removeTrimBackground(img, length = 120):
  img = img[:,length:-1*length]
  return img

def resizeScale(img, scale = 30):
  scale_percent = scale # percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)

  # resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  return resized

def preprocessingImg(img, dsize = (128,128)):
  # img = human_detection(img)
  img = removeTrimBackground(img, LENGTH_TRIM)
  img = resizeImage(img, dsize)
  return img

class MyVideo:
  def __init__(self, path='', root_path=''):
    self.path = path
    self.root = root_path
    self.video = cv2.VideoCapture(path)
    self.name = path.split("/")[-1]
    self.label = path.split("/")[-2]
  def getVideo(self):
    return self.video
  def getFullPath(self):
    return self.root + '/' + self.path
  def getFileName(self):
    return self.name
  def getFrameCount(self):
    return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
  def getLabel(self):
    return self.label
  def release(self):
    self.video.release()
    del self


def videotoframe(myVideo, dsize=(128,128) , numframe=5, start=0, stop=0):
  listframe = []
  length = myVideo.getFrameCount()

  currentFrame = -1
  skipframe = int((length - start - stop)/numframe)
  takeframe = skipframe + start
  taken = 0
  # print('video',video,'length', length, 'skip', skipframe)
  video = myVideo.getVideo()
  while(True):
      # Capture frame-by-frame
      ret, frame = video.read()
      currentFrame += 1

      if ret == False:
          break
      if currentFrame < takeframe:
          continue
      if taken >= numframe:
          break
      frame = resizeScale(frame)
      frame = removeTrimBackground(frame, LENGTH_TRIM)
      frame = resizeImage(frame, dsize)
      listframe.append(frame)

      takeframe += skipframe
      taken += 1
  # When everything done, release the capture
  myVideo.release()
  cv2.destroyAllWindows()
  return listframe

def preprocessingVideo(myVideo):
  frames = videotoframe(myVideo, SIZE, NBFRAME, 10, 30)
  frames = np.array(frames)
  frames = frames / 255.
  return frames

def predictOneVideo(video, classes):
  # preprocessing video path
  if type(video) is not MyVideo:
    video = MyVideo(video)
  item = preprocessingVideo(video)
  item = item[None,:]  # [640,10,64,64,3]
  try:
    # a = model.predict_classes(item) # old version
    predicted = np.argmax(model.predict(item), axis=-1) # [10] [15,12]
    return classes[predicted[0]]  # [16] accept
  except:
    print('Video', myVideo.name,'has error')
    return 'Null'
  # print(predicted)

def preparePath(path, csv_file = ''):
  paths = []
  if csv_file == '':
    videos = os.listdir(path)
    for video in videos:
        video_path = os.path.join(path, video)
        paths.append(video_path)
  else:
    with open(csv_file) as csvfile: # read path from csv
      reader = csv.reader(csvfile)
      for row in reader:
        video_path = os.path.join(path, row[0])
        paths.append(video_path)

  print(len(paths))
  return paths

def predictVideoOneTime(paths, classes, verbose = 0):
  data = []
  names = []
  expecteds = []
  n = len(videos)
  for path in paths:
    myVideo = MyVideo(path)
    item = preprocessingVideo(myVideo)
    data.append(item)
    names.append(myVideo.getFileName())
    expecteds.append(myVideo.getLabel())

  data = np.array(data)
  # print(data.shape)
  predicted = np.argmax(model.predict(data), axis=-1)
  # print(predicted)
  # print(classes[predicted])
  dic = {}
  count = 0
  for i in range(n):
    expected = expecteds[i]
    text = 'Video: '+ names[i] + ' Expected: '+ expected + ' Predicted: '+ classes[predicted[i]]
    if expected not in dic:
      dic[expected] = 0
    if expected == classes[predicted[i]]:
      count = count + 1
      dic[expected] = dic[expected] + 1
    else:
      text = '\x1b[31m'+ text + '\x1b[0m'

    if verbose == 1 :
      print(text)
  print()
  print('Accuray:',str(count)+'/'+str(n), ',',str(count/n))
  return dic

def predictVideoOneByOne(paths, classes, verbose = 0):
  dic = {}

  count = 0
  n = len(paths)
  index = 1
  for video in paths:
    myVideo = MyVideo(video)
    predicted = predictOneVideo( myVideo, classes)
    expected = myVideo.getLabel()
    print(index, end=' ')
    text = 'Video: '+ myVideo.getFileName() + ' Expected: '+  expected + ' Predicted: '+ predicted
    if expected not in dic:
      dic[expected] = 0
    if expected == predicted:
      count = count + 1
      dic[expected] = dic[expected] + 1
    else:
      text = '\x1b[31m'+ text + '\x1b[0m'
    index = index + 1

    if verbose == 1 :
      print(text)

  print()
  print('Accuray:',str(count)+'/'+str(n), ',',str(count/n))
  return dic

def predictVideoOneByOneReturn2dDict(paths, classes, verbose = 0):
  dic = {}    # to make a confusion maxtrix, im use a 2D dictionary
  count = 0
  n = len(paths)
  index = 0
  for video in paths:
    myVideo = MyVideo(video)
    predicted = predictOneVideo( myVideo, classes)
    expected = myVideo.getLabel()
    print(index+1, end=' ')
    text = 'Video: '+ myVideo.getFileName() + ' Expected: '+  expected + ' Predicted: '+ predicted
    if expected not in dic:
      dic[expected] = dict()
    if predicted not in dic[expected]:
      dic[expected][predicted] = 0
    if expected == predicted:
      count = count + 1
      # dic[expected] = dic[expected] + 1
    else:
      text = '\x1b[31m'+ text + '\x1b[0m'

    # process onfusion matrix array
    dic[expected][predicted] += 1


    index = index + 1
    if verbose == 1 :
      print(text)

  print()
  print('Accuray:',str(count)+'/'+str(n), ',',str(count/n))
  return dic

def predictVideo(path, csv_file, classes, verbose = 0 ,isOneTime = False):
  paths = preparePath(path, csv_file)

  if isOneTime == True:
    dic = predictVideoOneTime(paths, classes, verbose)
  else:
    dic = predictVideoOneByOneReturn2dDict(paths, classes, verbose)
  return dic

def load_VGG16_model(shape=(112, 112, 3)):
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)
  print("Model loaded..!")
  base_model.summary()

  # model.add(Flatten())
  output1 = GlobalMaxPool2D()
  output2 = Flatten()
  module = keras.Sequential([base_model, output1, output2])
  module.summary()
  return module

def LSTMModel(shape=(5, 112, 112, 3), nbout=3):
  model = Sequential()
  convnet = load_VGG16_model(shape[1:])
  # convnet = VGG16_small(shape[1:])
  model.add(TimeDistributed(convnet, input_shape=shape))

  model.add(LSTM(256, dropout=0.2))
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(nbout, activation='softmax'))
  sgd = keras.optimizers.SGD(lr=0.00005, decay = 1e-6, momentum=0.9, nesterov=True)
  # sgd = keras.optimizers.SGD( momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# read classes from file
classes_file = 'classes.npy'
encoder = LabelEncoder()
encoder.classes_ = np.load(classes_file)
classes = np.array(encoder.classes_)
num_classes = len(classes)
INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (2560, 5, 112, 112, 3)
print(classes)
# print(INSHAPE)
model = LSTMModel(INSHAPE, num_classes)
model.load_weights('chkp/weights-LSTM-095625-08_05.hdf5')

print(num_classes,classes)
