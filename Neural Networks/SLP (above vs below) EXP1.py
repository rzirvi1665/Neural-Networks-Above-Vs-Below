import tensorflow as tf
from tensorflow import keras
from createData import *
from utilityFunctions import *
from sklearn.utils import shuffle

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

trainingData=reshapeMLP(trainingData, numTrain)
testingData=reshapeMLP(testingData, numTest)

trainingLabels = storeLabels(numTrain)
testingLabels = storeLabels(numTest)

showRandomImage(trainingData, trainingLabels, numTrain)

print("Before Shuffle: ", end=" ")
checkShuffle(trainingLabels)

trainingData, trainingLabels = shuffle(trainingData, trainingLabels)
# testingData, testingLabels=shuffle(testingData, testingLabels)

print("After Shuffle: ", end=" ")
checkShuffle(trainingLabels)


trainingData=preprocess(trainingData)
testingData=preprocess(testingData)

# activation="sigmoid", loss="binary_crossentropy", 1 output neuron
# activation="softmax", loss="sparse_categorical_crossentropy", 2 output neurons

model = keras.Sequential([  # usually sequential because layers
	keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
	keras.layers.Dense(2, activation="softmax")  # output layer (3)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingData, trainingLabels, epochs=10)  # more epochs is not always better

model.summary()

for finalWeights in model.trainable_variables:
	if finalWeights.shape==(784, 2):
		heatMapSLP(finalWeights)
		break

predictions = model.predict(testingData)

# True tests that network can pick the "above the bar", False tests that network can pick "below" the bar image

accuracyAbove=testData(predictions, numTest, True, testingData, testingLabels)
accuracyBelow=testData(predictions, numTest, False, testingData, testingLabels)

print("Testing accuracy (choosing above image): " + str(accuracyAbove))
print("Testing accuracy (choosing below image): " + str(accuracyBelow))

testLoss, testAccuracy = model.evaluate(testingData, testingLabels)

predictCategorical(predictions, testingLabels, numTest, 5)
