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

XDist, YDist, EuclidDist = createEXP2DistanceAnalysis(numTrain, numTest)

print("Done generating images")
print("Currently appending data...")

trainingData = appendData(trainingData, numTrain, 0, 2)
testingData = appendData(testingData, numTest, numTrain, 2)

trainingData = reshapeMLP(trainingData, numTrain)
testingData = reshapeMLP(testingData, numTest)

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

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28, 28)),
	keras.layers.Dense(256, activation="relu"),
	keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(trainingData, trainingLabels, epochs=10)

model.summary()

predictions = model.predict(testingData)

accuracyAbove = testData(predictions, numTest, True, testingData, testingLabels)
accuracyBelow = testData(predictions, numTest, False, testingData, testingLabels)

print("Testing accuracy (choosing above image): " + str(accuracyAbove))
print("Testing accuracy (choosing below image): " + str(accuracyBelow))

testLoss, testAccuracy = model.evaluate(testingData, testingLabels)

predictCategorical(predictions, testingLabels, numTest, 5)

XDistCorrect = []
YDistCorrect = []
EuclidDistCorrect = []

XDistWrong = []
YDistWrong = []
EuclidDistWrong = []

count=0

for i in range(2*numTest):
	if np.argmax(predictions[i])!=testingLabels[i]:
		count+=1

for i in range(2*numTest):
	if np.argmax(predictions[i])==testingLabels[i]:
		if len(XDistCorrect)<count:
			XDistCorrect.append(XDist[i])
			YDistCorrect.append(YDist[i])
			EuclidDistCorrect.append(EuclidDist[i])
	else:
		XDistWrong.append(XDist[i])
		YDistWrong.append(YDist[i])
		EuclidDistWrong.append(EuclidDist[i])

print(len(XDistCorrect))
print(len(XDistWrong))
	
overlapHistogram("X-Distance", XDistCorrect, XDistWrong)
overlapHistogram("Y-Distance", YDistCorrect, YDistWrong)
overlapHistogram("Euclidean-Distance", EuclidDistCorrect, EuclidDistWrong)
