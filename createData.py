from PIL import Image
import random
import numpy as np

def triangle(pixelsAbove, pixelsBelow, height, x_startAbove, x_startBelow, upsideDown, distance):
  # ideally, height of 6 = 36 pixels
  largestRow=1
  for i in range(height-1):
    largestRow+=2

  aboveUpperLeftX = random.randint(1, 27 - largestRow)
  aboveUpperLeftY = random.randint(1, x_startAbove - distance - height)

  belowUpperLeftX = random.randint(1, 27 - largestRow)
  belowUpperLeftY = random.randint(x_startBelow + distance+1, 27 - height)

  if upsideDown:
    start=aboveUpperLeftX
    end=aboveUpperLeftX+largestRow-1
    for i in range (aboveUpperLeftY, aboveUpperLeftY+height):
      for j in range(start, end+1):
        pixelsAbove[j, i]=0
      start+=1
      end-=1

    start = belowUpperLeftX
    end = belowUpperLeftX + largestRow-1
    for i in range(belowUpperLeftY, belowUpperLeftY + height):
      for j in range(start, end+1):
        pixelsBelow[j, i] = 0
      start += 1
      end -= 1

  else:
    start=int((aboveUpperLeftX+aboveUpperLeftX+largestRow-1) / 2)
    end = int((aboveUpperLeftX + aboveUpperLeftX + largestRow-1) / 2)
    for i in range (aboveUpperLeftY, aboveUpperLeftY+height):
      for j in range(start, end+1):
        pixelsAbove[j, i]=0
      start-=1
      end+=1

    start = int((belowUpperLeftX + belowUpperLeftX + largestRow-1) / 2)
    end = int((belowUpperLeftX + belowUpperLeftX + largestRow-1) / 2)
    for i in range (belowUpperLeftY, belowUpperLeftY+height):
      for j in range(start, end+1):
        pixelsBelow[j, i]=0
      start-=1
      end+=1


def diamond(pixelsAbove, pixelsBelow, height, x_startAbove, x_startBelow, distance):
  # Use height of 7
  largestRow = 1
  for i in range(int((height - 1)/2)):
    largestRow += 2

  aboveUpperLeftX = random.randint(1, 27 - largestRow)
  aboveUpperLeftY = random.randint(1, x_startAbove - distance - height)

  belowUpperLeftX = random.randint(1, 27 - largestRow)
  belowUpperLeftY = random.randint(x_startBelow + distance+1, 27 - height)

  narrow=False
  start = int((aboveUpperLeftX + aboveUpperLeftX + largestRow - 1) / 2)
  end = int((aboveUpperLeftX + aboveUpperLeftX + largestRow - 1) / 2)
  for i in range(aboveUpperLeftY, aboveUpperLeftY + height):
    for j in range(start, end + 1):
      pixelsAbove[j, i] = 0

    if end-start==largestRow-1:
      narrow=True
    if narrow:
      start+=1
      end-=1
    else:
      start -= 1
      end += 1

  narrow=False
  start = int((belowUpperLeftX + belowUpperLeftX + largestRow - 1) / 2)
  end = int((belowUpperLeftX + belowUpperLeftX + largestRow - 1) / 2)
  for i in range(belowUpperLeftY, belowUpperLeftY + height):
    for j in range(start, end + 1):
      pixelsBelow[j, i] = 0
    if end-start==largestRow-1:
      narrow=True
    if narrow:
      start+=1
      end-=1
    else:
      start -= 1
      end += 1

def circle(pixelsAbove, pixelsBelow, radius, x_startAbove, x_startBelow, distance):
  # Use radius 7
  aboveUpperLeftX = random.randint(1, 27 - radius)
  aboveUpperLeftY = random.randint(1, x_startAbove - distance - radius)

  belowUpperLeftX = random.randint(1, 27 - radius)
  belowUpperLeftY = random.randint(x_startBelow + distance+1, 27 - radius)

  count=0
  entered=False
  start = int((aboveUpperLeftX + aboveUpperLeftX + radius - 1) / 2)-1
  end = int((aboveUpperLeftX + aboveUpperLeftX + radius - 1) / 2)+1
  for i in range(aboveUpperLeftY, aboveUpperLeftY + radius):
    for j in range(start, end + 1):
      pixelsAbove[j, i] = 0

    if end - start == radius - 1:
      count += 1
      entered = True

    if count >= 3:
      start += 1
      end -= 1

    elif not entered:
      start -= 1
      end += 1

  count=0
  entered=False
  start = int((belowUpperLeftX + belowUpperLeftX + radius - 1) / 2)-1
  end = int((belowUpperLeftX + belowUpperLeftX + radius - 1) / 2)+1
  for i in range(belowUpperLeftY, belowUpperLeftY + radius):
    for j in range(start, end + 1):
      pixelsBelow[j, i] = 0

    if end - start == radius - 1:
      count += 1
      entered=True

    if count >= 3:
      start += 1
      end -= 1

    elif not entered:
      start -= 1
      end += 1

def rectangle(pixelsAbove, pixelsBelow, width, length, x_startBelow, x_startAbove, distance):
  # can be used for any size rectangle (9 by 4 or 6 by 6)

  aboveUpperLeftX = random.randint(1, 27 - width)
  aboveUpperLeftY = random.randint(1, x_startBelow - distance - length)

  belowUpperLeftX = random.randint(1, 27 - width)
  belowUpperLeftY = random.randint(x_startAbove + distance + 1, 27 - length)

  for i in range(aboveUpperLeftX, aboveUpperLeftX+width):
    for j in range (aboveUpperLeftY, aboveUpperLeftY+length):
      pixelsAbove[i,j]=0

  for i in range(belowUpperLeftX, belowUpperLeftX+width):
    for j in range(belowUpperLeftY, belowUpperLeftY+length):
      pixelsBelow[i,j]=0
      
def referentCircleEXP2(upperLeftX, upperLeftY, pixels, radius):
  # Use radius 7
  
  count=0
  entered=False
  start = int((upperLeftX + upperLeftX + radius - 1) / 2)-1
  end = int((upperLeftX + upperLeftX + radius - 1) / 2)+1
  for i in range(upperLeftY, upperLeftY + radius):
    for j in range(start, end + 1):
      pixels[j, i] = 0

    if end - start == radius - 1:
      count += 1
      entered = True

    if count >= 3:
      start += 1
      end -= 1

    elif not entered:
      start -= 1
      end += 1



def storeLabels(numImages):

  labels=[]

  for i in range(numImages):
    labels.append(1)
  for i in range(numImages):
    labels.append(0)

  return np.array(labels)



def createEXP1(numTrain, numTest):

  # distance= 2 means not touching line
  # distance= -3 means can partly be on other side
  # dustance= -9 means shape can be completely on other side
  startShapeTrain= 1
  endShapeTrain= 5
  distanceTrain = 2
  
  startShapeTest = 6
  endShapeTest = 6
  distanceTest = 2
  
  maxEach = maxEachTypeOfShape(numTrain, startShapeTrain, endShapeTrain)

  createEachSetEXP1(maxEach, numTrain, 0, distanceTrain, startShapeTrain, endShapeTrain)

  maxEach = maxEachTypeOfShape(numTest, startShapeTest, endShapeTest)
  
  createEachSetEXP1(maxEach, numTest, numTrain, distanceTest, startShapeTest, endShapeTest)


def createEXP2(numTrain, numTest):
  # distance= 3
  
  startShapeTrain = 1
  endShapeTrain = 4
  distanceTrain = 3
  
  startShapeTest = 5
  endShapeTest = 5
  distanceTest = 3
  
  maxEach = maxEachTypeOfShape(numTrain, startShapeTrain, endShapeTrain)
  
  createEachSetEXP2(maxEach, numTrain, 0, distanceTrain, startShapeTrain, endShapeTrain)
  
  maxEach = maxEachTypeOfShape(numTest, startShapeTest, endShapeTest)
  
  createEachSetEXP2(maxEach, numTest, numTrain, distanceTest, startShapeTest, endShapeTest)


def maxEachTypeOfShape(numTrain, startShape, endShape):
  totalShapesPossible=endShape-startShape+1
  if numTrain % totalShapesPossible == 0:
    maxEach = int(numTrain / totalShapesPossible)
  else:
    maxEach = int(numTrain / totalShapesPossible) + 1
  return maxEach


def createEachSetEXP1(maxEach, numTrain, added, distance, startShape, endShape):
  shapeType = [0] * 6
  for i in range(numTrain):
    imgAbove = Image.new('L', (28, 28), color='white')
    imgBelow = Image.new('L', (28, 28), color='white')
    x_start=random.randint(13, 14)
    # x_start = random.randint(10, 17)
    pixelsAbove = imgAbove.load()  # pixel map
    pixelsBelow = imgBelow.load()
    width = 28
    for y in range(width):
      pixelsAbove[y, x_start - 1] = 0
      pixelsAbove[y, x_start] = 0
      pixelsAbove[y, x_start + 1] = 0
      pixelsBelow[y, x_start - 1] = 0
      pixelsBelow[y, x_start] = 0
      pixelsBelow[y, x_start + 1] = 0
    
    while True:
      shape = random.randint(startShape, endShape)
      if shapeType[shape - 1] < maxEach:
        break
    
    # Normal Distance (last parameter) = 2
    if shape == 1:
      # 6 by 6 Square
      rectangle(pixelsAbove, pixelsBelow, 6, 6, x_start, x_start, distance)
      shapeType[0] += 1
    
    elif shape == 2:
      # 9 by 4 Rectangle
      rectangle(pixelsAbove, pixelsBelow, 9, 4, x_start, x_start, distance)
      shapeType[1] += 1
    
    elif shape == 3:
      # Regular Triangle with height 6
      triangle(pixelsAbove, pixelsBelow, 6, x_start, x_start, False, distance)
      shapeType[2] += 1
    
    elif shape == 4:
      # Upside-down Triangle with height 6
      triangle(pixelsAbove, pixelsBelow, 6, x_start, x_start, True, distance)
      shapeType[3] += 1
    
    elif shape == 5:
      # Diamond with height 7, only use odd radius
      diamond(pixelsAbove, pixelsBelow, 7, x_start, x_start, distance)
      shapeType[4] += 1
    
    elif shape == 6:
      # Circle with radius 7, only use odd radius
      circle(pixelsAbove, pixelsBelow, 7, x_start, x_start, distance)
      shapeType[5] += 1

    imgAbove.save(f'imagesExperiment1/image({(i + 1 + added)}).png')
    imgBelow.save(f'imagesExperiment1/image({-(i + 1 + added)}).png')


def createEachSetEXP2(maxEach, numTrain, added, distance, startShape, endShape):
  shapeType = [0] * 6
  for i in range(numTrain):
    imgAbove = Image.new('L', (28, 28), color='white')
    imgBelow = Image.new('L', (28, 28), color='white')
    pixelsAbove = imgAbove.load()  # pixel map
    pixelsBelow = imgBelow.load()
    radius=7
    
    aboveUpperLeftX = random.randint(1, 27-radius)
    aboveUpperLeftY = random.randint(1, 27 - radius - 14)

    belowUpperLeftX = random.randint(1, 27 - radius)
    belowUpperLeftY = random.randint(14, 27 - radius)

  # for control 1, should be high accuracy
  #   if added!=0:
  #     aboveUpperLeftX = 10
  #     aboveUpperLeftY = 10
  #
  #     belowUpperLeftX = 10
  #     belowUpperLeftY = 10
    

    x_startAbove=aboveUpperLeftY+3
    x_startBelow=belowUpperLeftY+3
    
    referentCircleEXP2(belowUpperLeftX, belowUpperLeftY, pixelsAbove, radius)
    referentCircleEXP2(aboveUpperLeftX, aboveUpperLeftY, pixelsBelow, radius)
    
    # for control 2, should be low accuracy
    # if added==0:
    
    while True:
      shape = random.randint(startShape, endShape)
      if shapeType[shape - 1] < maxEach:
        break
  
    # Normal Distance (last parameter) = 2
    if shape == 1:
      # 6 by 6 Square
      rectangle(pixelsAbove, pixelsBelow, 6, 6, x_startBelow, x_startAbove, distance)
      shapeType[0] += 1
  
    elif shape == 2:
      # 9 by 4 Rectangle
      rectangle(pixelsAbove, pixelsBelow, 9, 4, x_startBelow, x_startAbove, distance)
      shapeType[1] += 1
  
    elif shape == 3:
      # Regular Triangle with height 6
      triangle(pixelsAbove, pixelsBelow, 6, x_startBelow, x_startAbove, False, distance)
      shapeType[2] += 1
  
    elif shape == 4:
      # Upside-down Triangle with height 6
      triangle(pixelsAbove, pixelsBelow, 6, x_startBelow, x_startAbove, True, distance)
      shapeType[3] += 1
  
    elif shape == 5:
      # Diamond with height 7, only use odd radius
      diamond(pixelsAbove, pixelsBelow, 7, x_startBelow, x_startAbove, distance)
      shapeType[4] += 1
  
    elif shape == 6:
      # Circle with radius 7, only use odd radius
      circle(pixelsAbove, pixelsBelow, 7, x_startBelow, x_startAbove, distance)
      shapeType[5] += 1
    
    imgAbove.save(f'imagesExperiment2/image({(i + 1 + added)}).png')
    imgBelow.save(f'imagesExperiment2/image({-(i + 1 + added)}).png')

    
    
    
    