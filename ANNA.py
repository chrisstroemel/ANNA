from ctypes import *
import matplotlib.pyplot as plotter
ANNA = cdll.LoadLibrary('./libANNA.so')
	
loggingInterval = 100


neuronInputLength = ANNA.getInputLength()
InputType = (neuronInputLength * c_float)
InputAddressType = (neuronInputLength * c_int)

ANNA.constructNetwork.argtypes = [c_int, c_int, c_float]
ANNA.constructNetwork.restype = c_void_p
#ANNA.setMaximizationPreference.argtypes = [c_float]
#ANNA.trainToConvergence.argtypes = [c_void_p, c_float, c_float]
#ANNA.trainToConvergence.restype = c_double
ANNA.stepNetwork.argtypes = [c_void_p, POINTER(c_float)]
ANNA.stepNetwork.restype = c_int
ANNA.runUntilOutput.argtypes = [c_void_p, POINTER(c_float)]
ANNA.runUntilOutput.restype = c_int
ANNA.setAction.argtypes = [c_void_p, c_int, c_int]
ANNA.setInputs.argtypes = [c_void_p, c_int, InputAddressType]
ANNA.reinforce.argtypes = [c_void_p, c_float]
ANNA.resetAccumulations.argtypes = [c_void_p]
ANNA.resetNetwork.argtypes = [c_void_p]
ANNA.deleteNetwork.argtypes = [c_void_p]

ANNA.getIterationsSinceReinforcement.argtypes = [c_void_p]
ANNA.getNeuronErrorRate.argtypes = [c_void_p, c_int]
ANNA.getNeuronErrorRate.restype = c_float
ANNA.getNeuronAccumulation.argtypes = [c_void_p, c_int]
ANNA.getNeuronAccumulation.restype = c_float

class Network:
	__actions = [(lambda : None)]

	def __init__(self, numberOfNeurons, inputLength, neuronAccumulationRate = 0.05, maxReinforcementHistory = 100000, loggingEnabled=False):
		self.__network = ANNA.constructNetwork(c_int(numberOfNeurons), c_int(maxReinforcementHistory), c_float(neuronAccumulationRate))
		self.numberOfNeurons = numberOfNeurons
		self.__inputLength = inputLength
		self.__loggingEnabled = loggingEnabled
		self.__iterationNumber = 0
		if loggingEnabled:
			self.__convergenceHistory = []
			self.__outputHistory = []
			self.__neuronErrorRates = [ [] for neuron in range(numberOfNeurons)]
			self.__neuronAccumulations = [ [] for neuron in range(numberOfNeurons)]
			self.__iterationLengthHistory = []
			self.__movingAverage = 1

	def __del__(self):
		ANNA.deleteNetwork(self.__network)

	def setAction(self, neuronNumber, action):
		#assert(action > 0)
		assert(neuronNumber >= 0 and neuronNumber < self.numberOfNeurons)
		actionCode = len(self.__actions)
		self.__actions.append(action)
		ANNA.setAction(self.__network, c_int(int(neuronNumber)), c_int(actionCode))
		self.__movingAverage = 1 / actionCode #initial value

	def setInputs(self, neuronNumber, inputAddresses):
		assert(len(inputAddresses) == neuronInputLength)
		for element in inputAddresses:
			assert(element >= 0 and element < self.__inputLength)
		ANNA.setInputs(self.__network, c_int(int(neuronNumber)), InputAddressType(*inputAddresses))
		
	def stepNetwork(self, inputVector):
		#assert(len(inputVector) == self.__inputLength)
		cArray = (c_float * len(inputVector))(*inputVector)
		actionCode = ANNA.stepNetwork(self.__network, pointer(cArray))
		return self.__actions[actionCode]()

	def runUntilOutput(self, inputVector):
		#assert(len(inputVector) == self.__inputLength)
		cArray = (c_float * len(inputVector))(*inputVector)
		actionCode = ANNA.runUntilOutput(self.__network, cArray)
		self.__iterationNumber += 1
		if self.__loggingEnabled and ((self.__iterationNumber % loggingInterval) == 0):
			elapsedIterations = ANNA.getIterationsSinceReinforcement(self.__network)
			self.__iterationLengthHistory.append(elapsedIterations)
			for neuron in range(self.numberOfNeurons):
				errorRate = 1 - ANNA.getNeuronErrorRate(self.__network, c_int(neuron))
				self.__neuronErrorRates[neuron].append(errorRate)
				accumulation = ANNA.getNeuronAccumulation(self.__network, c_int(neuron))
				self.__neuronAccumulations[neuron].append(accumulation)
		return self.__actions[actionCode]

	def reinforce(self, reinforcement):
		ANNA.reinforce(self.__network, c_float(reinforcement))

	def resetAccumulations(self):
		ANNA.resetAccumulations(self.__network)

	def resetNetwork(self):
		ANNA.resetNetwork(self.__network)

	def showPlots(self, errorHistory):
		figure0 = plotter.figure(0)
		plotter.plot(range(len(errorHistory)), errorHistory)
		
		if self.__loggingEnabled:
			iterations = self.__iterationLengthHistory
			xAxis = range(0, len(iterations) * loggingInterval, loggingInterval)
			figure1 = plotter.figure(1)
			plotter.scatter(xAxis, iterations, s=0.2)
			figure2 = plotter.figure(2)
			for neuron in range(self.numberOfNeurons):
				errorHistory = self.__neuronErrorRates[neuron]
				plotter.plot(xAxis, errorHistory, linewidth=1, label='Neuron #' + str(neuron))
			plotter.legend()
			figure3 = plotter.figure(3)
			for neuron in range(self.numberOfNeurons):
				accumulations = self.__neuronAccumulations[neuron]
				plotter.scatter(xAxis, accumulations, s=0.2)
			plotter.legend()
			plotter.show()
		
		









