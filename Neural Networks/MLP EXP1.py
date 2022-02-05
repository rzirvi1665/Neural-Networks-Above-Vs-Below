import tensorflow as tf
from tensorflow import keras
from createData import *
from utilityFunctions import *
from sklearn.utils import shuffle

trainingData = np.empty(0)
testingData = np.empty(0)

numTrain = 30000
numTest = 5000

print("Right about to generate images...")

createEXP1(numTrain, numTest)

print("Done generating images")
print("Currently appending data...")

trainingData = appendData(trainingData, numTrain, 0, 1)
testingData = appendData(testingData, numTest, numTrain, 1)

trainingData=reshapeMLP(trainingData, numTrain)
testingData=reshapeMLP(testingData, numTest)

trainingLabels = storeLabels(numTrain)
testingLabels = storeLabels(numTest)

showRandomImage(trainingData, trainingLabels, numTrain)

print("Before Shuffle: ", end=" ")
checkShuffle(trainingLabels)

trainingData, trainingLabels = shuffle(trainingData, trainingLabels)


print("After Shuffle: ", end=" ")
checkShuffle(trainingLabels)

trainingData=preprocess(trainingData)
testingData=preprocess(testingData)

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(256, activation="relu"),
  keras.layers.Dense(2, activation="softmax") 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingData, trainingLabels, epochs=10)

model.summary()

for finalWeights in model.trainable_variables:
  if finalWeights.shape==(784, 256):
      heatMapMLP(finalWeights)
      break

for finalWeights in model.trainable_variables:
  if finalWeights.shape == (784, 256):
    initialWeights = finalWeights
  if finalWeights.shape == (256, 2):
    heatMapMLPOutput(finalWeights, initialWeights, 1)
    break


predictions = model.predict(testingData)

accuracyAbove=testData(predictions, numTest, True, testingData, testingLabels)
accuracyBelow=testData(predictions, numTest, False, testingData, testingLabels)

print("Testing accuracy (choosing above image): " + str(accuracyAbove))
print("Testing accuracy (choosing below image): " + str(accuracyBelow))

testLoss, testAccuracy = model.evaluate(testingData, testingLabels)

predictCategorical(predictions, testingLabels, numTest, 5)
