import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def compare(firstImg, secondImg, pickAbove):
  
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
  norm = TwoSlopeNorm(vcenter=0)
  x = plt.imshow(heatMapImage, norm=norm, cmap="seismic", interpolation='nearest')
  
  plt.colorbar(x)
  plt.show()


def divideHeatMap(heatMapImage, type):
  norm = TwoSlopeNorm(vcenter=0)
  x = plt.imshow(heatMapImage, norm=norm, cmap="seismic", interpolation='nearest')
  
  ax = plt.gca()
  ax.set_xticks(np.arange(-.5, 448, 28));
  ax.set_yticks(np.arange(-.5, 448, 28));
  ax.set_xticklabels([]);
  ax.set_yticklabels([]);
  
  ax.grid(color='w', linestyle='-', linewidth=2)
  
  plt.colorbar(x)
  
  if type==1:
    plt.savefig("MLPExp1Large.jpg", dpi=1500)
  else:
    plt.savefig("MLPExp2Large.jpg", dpi=1500)
  
  plt.show()
  
def showHistogram(values):
  binwidth=.1
  small=-1.3
  large=1.3
  plt.hist(values, bins=np.arange(small, large + binwidth, binwidth))
  plt.xlabel("Weight of Connection")
  plt.ylabel("Count")
  plt.show()
  
def overlapHistogram(title, a, b):
  if title=="Euclidean-Distance":
    binwidth=.2
  else:
    binwidth=1
  small=min(min(a), min(b))-1
  large=max(max(a), max(b))+1
  bins = np.arange(small, large + binwidth, binwidth)
  plt.hist(a, bins, alpha=0.5, label='Correct')
  plt.hist(b, bins, alpha=0.5, label='Incorrect')
  plt.legend(loc='upper right')
  plt.xlabel(title)
  plt.ylabel("Count")
  plt.savefig(f"MLP {title}.jpg", dpi=1200)
  plt.show()

def heatMapMLP(variable):
  heatMapImage = np.zeros((28, 28), dtype=float)
  for x in range(28):
    for y in range(28):
      heatMapImage[x][y] = variable[(x * 28) + y][0]
      
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
      heatMapImage[x][y] = variable[(x * 28) + y][0]

  showHeatMap(heatMapImage)


def heatMapMLPOutput(variable, variable2, type):
  heatMapImage = np.zeros((16, 16), dtype=float)
  values = np.zeros(256)
  for x in range(16):
    for y in range(16):
      values[(x * 16) + y] = variable[(x * 16) + y][0] - variable[(x * 16) + y][1]

  index = np.argsort(values)
  
  showHistogram(values)
  
  for x in range(16):
    for y in range(16):
      heatMapImage[x][y] = values[index[(x * 16) + y]]
  
  showHeatMap(heatMapImage)

  largeHeatMap = np.zeros((448, 448), dtype=float)

  startX=0
  startY=0

  for x in range(256):
    
    currX=startX
    currY=startY

    for y in range(28):
      for z in range(28):
        largeHeatMap[currX][currY] = variable2[(y * 28) + z][index[x]]
        currY += 1
      currY=startY
      currX += 1

    startY+=28
    if startY==448:
      startY=0
      startX+=28

  divideHeatMap(largeHeatMap, type)
  
def heatMapCNNOutput(variable):
  heatMapImage = np.zeros((16, 16), dtype=float)
  values = np.zeros(256)
  for x in range(16):
    for y in range(16):
      values[(x * 16) + y]= variable[(x * 16) + y][0] - variable[(x * 16) + y][1]
      
  values.sort()
  
  showHistogram(values)
  
  for x in range(16):
    for y in range(16):
      heatMapImage[x][y] = values[(x * 16)+y]

  showHeatMap(heatMapImage)

def preprocess(data):
  return data / 255.0
  

def predictCategorical(predictions, labels, numTest, numPredictions):
  
  for i in range(numPredictions):
    randomImg=random.randint(0, (numTest*2)-1)
    print("Prediction: " + str(np.argmax(predictions[randomImg])))
    print("Actual: " + str(labels[randomImg]))
    
def showIncorrect(data, labels, predictions, numTest, numImages):
  count = 1
  for i in range((numTest*2)-1):
    if np.argmax(predictions[i]) != labels[i]:
      plt.figure()
      plt.title(f"Prediction: {np.argmax(predictions[i])}    Answer: {labels[i]}")
      plt.imshow(data[i], cmap="binary")
      plt.savefig(f"MLPWrong{count}.jpg")
      count+=1
      if count>numImages:
        break
  
  for i in range((numTest*2)-1, -1, -1):
    if np.argmax(predictions[i]) != labels[i]:
      plt.figure()
      plt.title(f"Prediction: {np.argmax(predictions[i])}    Answer: {labels[i]}")
      plt.imshow(data[i], cmap="binary")
      plt.savefig(f"MLPWrong{count}.jpg")
      count+=1
      if count>numImages*2:
        break

def showCorrect(data, labels, predictions, numTest, numImages):
  count = 1
  for i in range((numTest * 2) - 1):
    if np.argmax(predictions[i]) == labels[i]:
      plt.figure()
      plt.title(f"Prediction: {np.argmax(predictions[i])}    Answer: {labels[i]}")
      plt.imshow(data[i], cmap="binary")
      plt.savefig(f"MLPCorrect{count}.jpg")
      count += 1
    if count > numImages:
      break
      
  for i in range((numTest*2)-1, -1, -1):
    if np.argmax(predictions[i]) == labels[i]:
      plt.figure()
      plt.title(f"Prediction: {np.argmax(predictions[i])}    Answer: {labels[i]}")
      plt.imshow(data[i], cmap="binary")
      plt.savefig(f"MLPCorrect{count}.jpg")
      count += 1
    if count > numImages*2:
      break


def combineImages():
  
  fig = plt.figure(figsize=(20, 2))
  
  rows=2
  columns=10

  for i in range(1, 11, 1):
    img = mpimg.imread(f'MLPCorrect{i}.jpg')
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.axis('off')
    
  for i in range(1, 11, 1):
    img = mpimg.imread(f'MLPWrong{i}.jpg')
    fig.add_subplot(rows, columns, i+10)
    plt.imshow(img)
    plt.axis('off')
  
  plt.savefig(f"MLPCombined.jpg")


def predictBinary(predictions, labels, numTest, numPredictions):
  for i in range(numPredictions):
    randomImg = random.randint(0, (numTest * 2) - 1)
    if predictions[randomImg]>=.5:
      result=1
    else:
      result=0
    print("Prediction: " + str(result))
    print("Actual: " + str(labels[randomImg]))
