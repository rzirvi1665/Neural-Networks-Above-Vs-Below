import tensorflow as tf
from tensorflow import keras
import tensorflow
from createData import *
from utilityFunctions import *
from sklearn.utils import shuffle

model = tensorflow.keras.Sequential()
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Activation = tensorflow.keras.layers.Activation
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense

trainingData = np.empty(0)
testingData = np.empty(0)

numTrain = 5000
numTest = 1500

print("Right about to generate images...")

createEXP1(numTrain, numTest)

print("Done generating images")
print("Currently appending data...")

trainingData = appendData(trainingData, numTrain, 0, 1)
testingData = appendData(testingData, numTest, numTrain, 1)

trainingData = reshapeCNN(trainingData, numTrain)
testingData = reshapeCNN(testingData, numTest)

trainingLabels = storeLabels(numTrain)
testingLabels = storeLabels(numTest)

showRandomImage(trainingData, trainingLabels, numTrain)

print("Before Shuffle: ", end=" ")
checkShuffle(trainingLabels)

trainingData, trainingLabels = shuffle(trainingData, trainingLabels)

print("After Shuffle: ", end=" ")
checkShuffle(trainingLabels)

trainingData = preprocess(trainingData)
testingData = preprocess(testingData)

print(trainingData.shape)
# activation="sigmoid", loss="binary_crossentropy", 1 output neuron
# activation="softmax", loss="sparse_categorical_crossentropy", 2 output neurons

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(87, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingData, trainingLabels, epochs=10)  # more epochs is not always better

model.summary()

test = np.zeros((783, 128), dtype=float)

#display heatmap
for finalWeights in model.trainable_variables:
	if finalWeights.shape == (783, 128):
		heatMapCNN(finalWeights)
		break

predictions = model.predict(testingData)

# True tests that network can pick the "above the bar", False tests that network can pick "below" the bar image

accuracyAbove = testData(predictions, numTest, True, testingData, testingLabels)
accuracyBelow = testData(predictions, numTest, False, testingData, testingLabels)

print("Testing accuracy (choosing above image): " + str(accuracyAbove))
print("Testing accuracy (choosing below image): " + str(accuracyBelow))

testLoss, testAccuracy = model.evaluate(testingData, testingLabels)

predictCategorical(predictions, testingLabels, numTest, 5)

