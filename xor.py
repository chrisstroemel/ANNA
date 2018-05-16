import ANNA
import random

#increasing the moving average weight seems to make it converge faster?
LEARNING_RATE = 1#0.01 (try going higher to compensate for locally lower learning rates)
NUMBER_OF_NEURONS = 4
network = ANNA.Network(NUMBER_OF_NEURONS, 2, 0.5, loggingEnabled = True)
for i in range(NUMBER_OF_NEURONS):
	network.setInputs(i, [0, 1])
network.setAction(0, 1)
network.setAction(NUMBER_OF_NEURONS / 2, 2)

inputs = [[1, 1], [0, 0], [1, 0], [0, 1]]
outputs = [1, 1, 2, 2]

errorMovingAverage = 0.5

errorHistory = []

maxIterations = 500000
for epoch in range (maxIterations):
	index = random.randrange(0,4)
	output = network.runUntilOutput(inputs[index])
	reinforcement = LEARNING_RATE
	error = 0
	if (output != outputs[index]):
		#print(output, index)
		reinforcement = -LEARNING_RATE
		error = 1
	network.reinforce(reinforcement)
	network.resetAccumulations()
	errorMovingAverage = (0.001 * error) + (0.999 * errorMovingAverage)
	errorHistory.append(errorMovingAverage)
	if (epoch % ((maxIterations / 20)) == 0): print(epoch / maxIterations)

print('Loading plots...')
network.showPlots(errorHistory)

	
