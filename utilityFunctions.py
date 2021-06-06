import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compare(firstImg, secondImg, pickAbove):
  """
  this compares two images and returns its prediction
  :param firstImg: first Image
  :param secondImg: second Image
  :param pickAbove: if pick above or below
  :return: prediction (0- first image or 1- second image)
  """
  
  predictFirst=np.argmax(firstImg)
  predictSecond=np.argmax(secondImg)
  
  if predictFirst==predictSecond:
    maxFirst=np.max(firstImg)
    maxSecond=np.max(secondImg)
    
    if maxFirst>maxSecond:
      if (predictFirst==1 and pickAbove) or (predictFirst==0 and not pickAbove):
        return 0
      else:
        return 1
      
    else:
      if (predictSecond==1 and pickAbove) or (predictSecond==0 and not pickAbove):
        return 1
      else:
        return 0
  
  elif (predictFirst==1 and pickAbove) or (predictFirst==0 and not pickAbove):
    return 0
  else:
    return 1


def testData(predictions, numTest, pickAbove, testingData, testingLabels):
  """
  :param testingLabels:
  :param testingData:
  :param predictions: list of predictions for each image
  :param numTest: number of testing images
  :param pickAbove: if pickAbove==True, network tries to pick "above image", if pickAbove==False, network tries to pick "below" image
  :return: accuracy of model
  """
  
  count=0
  for i in range(numTest):
    above=predictions[i]
    below=predictions[i+numTest]
    randomizePos=random.randint(0, 1)
    
    if randomizePos==0:
      choice=compare(above, below, pickAbove)
      if (choice==0 and pickAbove) or (choice==1 and not pickAbove):
        count+=1
  
    else:
      choice=compare(below, above, pickAbove)
      if (choice==1 and pickAbove) or (choice==0 and not pickAbove):
        count+=1
  
  if pickAbove:
    print("Above image results: " + str(count) + "/" + str(numTest))
  else:
    print("Below image results: " + str(count) + "/" + str(numTest))
    
  return count / float(numTest)


def createImage(i, numTest, testingData, testingLabels):
  plt.figure()
  plt.title(f"The answer for the dataset is {testingLabels[i+numTest]}")
  plt.imshow(testingData[i+numTest], cmap="binary")
  plt.colorbar()
  plt.show()



def appendData(data, numImages, startImage, EXP):
  if EXP==1:
    data = appendEachEXP1(data, numImages, startImage, True)
    data= appendEachEXP1(data, numImages, startImage, False)
  else:
    data = appendEachEXP2(data, numImages, startImage, True)
    data = appendEachEXP2(data, numImages, startImage, False)
  return data


def appendEachEXP1(data, numImages, startImage, sign):
  for i in range(numImages):
    if sign:
      img = Image.open(f'imagesExperiment1/image({i + 1 + startImage}).png')
    else:
      img = Image.open(f'imagesExperiment1/image({-(i + 1 + startImage)}).png')
    
    currImage = np.asarray(img)
    data = np.append(data, currImage)
    
  return data


def appendEachEXP2(data, numImages, startImage, sign):
  for i in range(numImages):
    if sign:
      img = Image.open(f'imagesExperiment2/image({i + 1 + startImage}).png')
    else:
      img = Image.open(f'imagesExperiment2/image({-(i + 1 + startImage)}).png')
    
    currImage = np.asarray(img)
    data = np.append(data, currImage)
  
  return data


def reshapeCNN(data, numImages):
  data = np.reshape(data, (2 * numImages, 28, 28, 1))
  print("Shape: " + str(data.shape))
  return data

def reshapeMLP(data, numImages):
  data = np.reshape(data, (2 * numImages, 28, 28))
  print("Shape: " + str(data.shape))
  return data


def checkShuffle(labels):
  for i in range(10):
    print(labels[i], end=" ")
  print()
  
  
    
def showRandomImage(data, labels, numImages):
  n = random.randint(0, (numImages * 2) - 1)
  plt.figure()
  plt.title(f"The answer for the {(n+1):,}th dataset is {labels[n]}")
  plt.imshow(data[n], cmap="binary")
  plt.colorbar()
  plt.show()

def showHeatMap(heatMapImage):
  plt.imshow(heatMapImage, cmap="seismic", interpolation='nearest')
  plt.colorbar()
  plt.show()

def heatMapMLP(variable):
  heatMapImage = np.zeros((28, 28), dtype=float)
  for x in range(28):
    for y in range(28):
      sum = 0
      for z in range(128):
        sum += variable[(x * 28) + y][z]
      heatMapImage[x][y] = sum
      
  showHeatMap(heatMapImage)


def heatMapCNN(variable):
  heatMapImage = np.zeros((28, 28), dtype=float)
  for x in range(28):
    for y in range(28):
      sum = 0
      if (x * 28) + y < 783:
        for z in range(128):
            sum += variable[(x * 28) + y][z]
        heatMapImage[x][y] = sum

  showHeatMap(heatMapImage)


def heatMapSLP(variable):
  heatMapImage = np.zeros((28, 28), dtype=float)
  for x in range(28):
    for y in range(28):
      sum = 0
      for z in range(2):
        sum += variable[(x * 28) + y][z]
      heatMapImage[x][y] = sum

  showHeatMap(heatMapImage)


def preprocess(data):
  return data / 255.0
  

def predictCategorical(predictions, labels, numTest, numPredictions):
  
  for i in range(numPredictions):
    randomImg=random.randint(0, (numTest*2)-1)
    print("Prediction: " + str(np.argmax(predictions[randomImg])))
    print("Actual: " + str(labels[randomImg]))


def predictBinary(predictions, labels, numTest, numPredictions):
  for i in range(numPredictions):
    randomImg = random.randint(0, (numTest * 2) - 1)
    if predictions[randomImg]>=.5:
      result=1
    else:
      result=0
    print("Prediction: " + str(result))
    print("Actual: " + str(labels[randomImg]))

