

#include "ANNA.h"
#define RANDOM_FLOAT ((float) rand() / (float) RAND_MAX)
#define NEURON_COOLING_RATE 1.0 // might need to be smaller
//#define MAXIMIZATION_PREFERENCE 1

#define MOVING_AVERAGE_WEIGHT 0.001 //0.001 //currently disabled in updateWeights()
#define INTERNAL_LEARNING_RATE 0.001
#define GRADIENT_PERSISTENCE 0.9
#define SQUARE_PERSISTENCE 0.999

float loopIterations = 0;

bool INITIALIZED = false;

Network* constructNetwork(int numberOfNeurons, int maxReinforcementHistory, float neuronAccumulationRate) {
	if (!INITIALIZED) initialize();
	if (numberOfNeurons < pow(2, (NUMBER_OF_CONNECTIONS / 2))) {
		printf("ERROR: network connectivity exceeds the size of the network");
		exit(1);
	}
	Network* network = calloc(1, sizeof(Network));
	network->currentIteration = 0;
	network->mostRecentReinforcement = 0;
	network->networkSize = numberOfNeurons;
	network->neuronAccumulationRate = neuronAccumulationRate;
	network->neurons = calloc(numberOfNeurons, sizeof(Neuron));
	network->currentNeuron = &(network->neurons[0]);
	for (int i = 0; i < numberOfNeurons; i++) {
		Neuron *neuron = &(network->neurons[i]);
		neuron->lastUpdated = 0;
		neuron->timesUsedSinceReinforcement = 0;
		neuron->cumulativeValue = 0.0;
		neuron->externalInputIndices[0] = 0;
		neuron->externalInputIndices[1] = 1;
		//neuron->externalInputIndices[2] = 2;
		neuron->actionCode = 0;
		neuron->connectionIndices[0] = i;
		neuron->localLearningRate = 1;
		neuron->timesReinforced = 1;
		for (int j = 0; j < (NUMBER_OF_CONNECTIONS / 2); j++) {
			int offset = pow(NETWORK_NONLOCALITY, j);
			//printf("Offset: %i \n", offset);
			//printf("Degree: %i\n", j);
			//printf("%i\n", (NUMBER_OF_CONNECTIONS / 10));
			neuron->connectionIndices[(j * 2) + 1] = wrapIndex(i + offset, numberOfNeurons);
			neuron->connectionIndices[(j * 2) + 2] = wrapIndex(i - offset, numberOfNeurons);
			//printf("Index: %i\n", neuron->connectionIndices[10]);
		}
		for (int j = 0; j < NUMBER_OF_CONNECTIONS; j++) {
			Synapse *synapse = &(neuron->synapses[j]);
			synapse->localLearningRate = 1.0 + (0.0 * (RANDOM_FLOAT - 0.5));
			synapse->amplitudeWeight[0] = 1.0 + (j * 10.0 / NUMBER_OF_CONNECTIONS);
			printf("%f \n", synapse->amplitudeWeight[0]);
			for (int k = 1; k < 4; k++) synapse->amplitudeWeight[k] = 0;
			for (int k = 0; k < WEIGHT_LENGTH; k++) {
				synapse->offsetWeights[0][k] = 0 * (RANDOM_FLOAT - 0.5);//zeros;
				synapse->periodWeights[0][k] = 2 + (0.0 * (RANDOM_FLOAT - 0.5));//ones;
				synapse->tightnessWeights[0][k] = 1 + (0.0 * (RANDOM_FLOAT - 0.5));
				for (int l = 1; l < 4; l++) {
					synapse->offsetWeights[l][k] = 0;
					synapse->periodWeights[l][k] = 0;
					synapse->tightnessWeights[l][k] = 0;
				}
			}
		}
		//printf("Neuron complete!\n");
	}
	network->currentIteration = 0;
	network->mostRecentReinforcement = 0;
	network->allocatedReinforcementHistoryLength = maxReinforcementHistory;
	network->numberOfReinforcements = 0;
	network->feedbackHistory = calloc(maxReinforcementHistory, sizeof(Reinforcement));
	return network;
}


int runUntilOutput(Network* network, float input[NUMBER_OF_INPUTS]) {
	//printf("%f %f\n", input[0], input[1]);
	int output = 0;
	unsigned long iterations = 0;
	while (!output) {
		output = stepNetwork(network, input);
		iterations++;
	}
	loopIterations += iterations;
	return output;
}



void initialize() {
	INITIALIZED = true;
	srand(time(NULL));
	//assert(lengthof(network) == networkSize);
	//if (RUN_UNIT_TEST) runMiscellaneousTests();	
}

int wrapIndex(int index, int size) {
	int wrappedIndex =  ((index % size) + size) % size;
	assert(wrappedIndex >= 0);
	assert(wrappedIndex < size);
	return wrappedIndex;
}

float cool(Neuron* neuron, unsigned long currentIteration) {
	float elapsedIterations = currentIteration - neuron->lastUpdated;
	float cooledValue = neuron->cumulativeValue / ((elapsedIterations * NEURON_COOLING_RATE) + 1);
	assert(!isnan(cooledValue));
	//printf("%f %f %f\n", cooledValue, neuron->cumulativeValue, elapsedIterations);
	assert(cooledValue < neuron->cumulativeValue || elapsedIterations == 0 || neuron->cumulativeValue == 0);
	return cooledValue;
}

void getInputs(Network* network, float *rawInputs, Neuron *neuron, float *combinedInputVector) {
	for (int i = 0; i < NUMBER_OF_INPUTS; i++) {
		combinedInputVector[i] = rawInputs[neuron->externalInputIndices[i]];
	}
	unsigned long currentIteration = network->currentIteration;
	for (int i = 0; i < NUMBER_OF_CONNECTIONS; i++) { // recurrent cooling
		combinedInputVector[i + NUMBER_OF_INPUTS] = logf(1 + cool(&(network->neurons[neuron->connectionIndices[i]]), currentIteration));
	}
	combinedInputVector[WEIGHT_LENGTH - 1] = logf(1 + cool(neuron, currentIteration));
}

// returns an int that can be interpeted as some special action that was assigned to this neuron
int stepNetwork(Network* network, float rawInputs[NUMBER_OF_INPUTS]/*, Network *network*/) {
	network->currentIteration++;
	//printf("Iteration number %i\n ", (int) network->currentIteration);
	Neuron* currentNeuron = network->currentNeuron;
	if (network->mostRecentReinforcement > currentNeuron->lastUpdated && currentNeuron->timesUsedSinceReinforcement > 0) {
		updateWeights(currentNeuron, getReinforcement(network, currentNeuron->lastUpdated));
	}
	//float activationSum = 0;
	float inputData[WEIGHT_LENGTH];
	getInputs(network, rawInputs, network->currentNeuron, inputData); // getInputs 
	int activationIndex = processNeuron(currentNeuron, inputData);
	assert(activationIndex < NUMBER_OF_CONNECTIONS);
	currentNeuron->timesUsedSinceReinforcement++;
	currentNeuron->lastUpdated = network->currentIteration;
	currentNeuron->cumulativeValue = /*1;*/network->neuronAccumulationRate + cool(currentNeuron, network->currentIteration);
	int actionCode = currentNeuron->actionCode;
	network->currentNeuron = &(network->neurons[currentNeuron->connectionIndices[activationIndex]]);
	return actionCode;
}

void setAction(Network* network, int index, int actionCode) {
	network->neurons[index].actionCode = actionCode;
}


// add to gradient sum on the first pass, and also store gradients
// then select the activated synapse
// then correct the just the selected synapse with the stored gradients

int processNeuron(Neuron* neuron, float* inputs) {
	float differences[NUMBER_OF_CONNECTIONS][WEIGHT_LENGTH];
	float modulusFactors[NUMBER_OF_CONNECTIONS][WEIGHT_LENGTH];
	float cumulativeActivation[NUMBER_OF_CONNECTIONS];
	float activations[NUMBER_OF_CONNECTIONS];
	float activationSum = 0;
	for (int i = 0; i < NUMBER_OF_CONNECTIONS; i++) {
		Synapse* synapse = &(neuron->synapses[i]);
		float exponentSum = 0;
		for (int j = 0; j < WEIGHT_LENGTH; j++) {
			float difference = inputs[j] - synapse->offsetWeights[0][j];
			differences[i][j] = difference;
			float period = synapse->periodWeights[0][j];
			//float halfPeriod = 0.5 * period;
			//float modulusFactor = fabsf(fmodf(difference + halfPeriod, period)) - halfPeriod;
			float modulusFactor = remainder(difference, period);
			//printf("%f %f %f %f %f\n", difference, inputs[j],synapse->offsetWeights[j], modulusFactor, period);
			assert(abs(modulusFactor) <= period / 2);
			//printf("Modulus factor: %f %f %f\n", modulusFactor, period, difference);
			modulusFactors[i][j] = modulusFactor;
			exponentSum += modulusFactor * modulusFactor * synapse->tightnessWeights[0][j];
		}
		float activation = exp(-exponentSum);
		float weightedActivation = synapse->amplitudeWeight[0] * activation;

		//printf("Amplitide weight: %f\n", synapse->amplitudeWeight);
		//printf("Activation: %f\n\n", activation);
		assert(activation > 0);
		assert(isnormal(activation));
		activationSum += weightedActivation + FLT_EPSILON;
		activations[i] = weightedActivation;
		cumulativeActivation[i] = activationSum;
	}
	float random = RANDOM_FLOAT * activationSum;
	//printf("Random: %f\n", random);
	//printf("Activation sum: %f", activationSum);


	int activationIndex = 0;
	while (random > cumulativeActivation[activationIndex]) activationIndex++;
	float localLearningRate = neuron->synapses[activationIndex].localLearningRate;
	//printf("%f\n", activations[activationIndex]);	

	// branch free
	for (int i = 0; i < NUMBER_OF_CONNECTIONS; i++) {
		float inverseActivationShare = activationSum / activations[i];
		//assert(isnormal(inverseActivationShare));
		// compute full boolean expression to remove data dependancy between loop iterations
		bool isActivatedIndex = (i == activationIndex);//(random > cumulativeActivation[i]) & (random < cumulativeActivation[i + 1]);
		float gain = (isActivatedIndex) ? localLearningRate : -localLearningRate;//1/*inverseActivationShare*/ : -1.0/*-1*/;
		//activationIndex = (isActivatedIndex) ? i : activationIndex;
		assert(i < lengthof(cumulativeActivation));
		Synapse* synapse = &(neuron->synapses[i]);
		float gradientFactor = gain * activations[i];
		//printf("\nGradient factor: %f\n", gradientFactor);
		assert(!isnan(gradientFactor));
		//synapse->amplitudeWeight[1] += gradientFactor / synapse->amplitudeWeight[1];
		for (int j = 0; j < WEIGHT_LENGTH; j++) {
			float modulusFactor = modulusFactors[i][j];
			synapse->tightnessWeights[1][j] += modulusFactor * modulusFactor * (-gradientFactor);
			float offsetGradient = 2 * synapse->tightnessWeights[0][j] * modulusFactor * gradientFactor;
			synapse->offsetWeights[1][j] += offsetGradient;
			assert(!isnan(synapse->offsetWeights[1][j]));
			assert(isfinite(synapse->offsetWeights[1][j]));
			synapse->periodWeights[1][j] += round(differences[i][j] / synapse->periodWeights[0][j]) * offsetGradient;
		}
	}
	//printf("Activated index: %i\n", activationIndex);
	assert(activationIndex > -1);
	assert(activationIndex < NUMBER_OF_CONNECTIONS);
	return activationIndex;
}

void updateWeights(Neuron *neuron, float reinforcement) {
	float normalizedReinforcement = /*neuron->localLearningRate * */reinforcement / (neuron->timesUsedSinceReinforcement);
	assert(!isnan(normalizedReinforcement));
	assert(isfinite(normalizedReinforcement));
	for (int i = 0; i < NUMBER_OF_CONNECTIONS; i++) {
		Synapse *synapse = &(neuron->synapses[i]);
		//float newAmplitudeWeight = synapse->amplitudeWeight[0] + (synapse->amplitudeWeight[1] * normalizedReinforcement);
		//synapse->amplitudeWeight[0] = clamp(newAmplitudeWeight, FLT_EPSILON, 1);
		//synapse->amplitudeWeight[1] = 0;

		unsigned long reinforcementNumber = neuron->timesReinforced;
		applyADAM(synapse->offsetWeights, normalizedReinforcement, reinforcementNumber, -20, 20);
		applyADAM(synapse->periodWeights, normalizedReinforcement, reinforcementNumber, FLT_EPSILON, 20);
		applyADAM(synapse->tightnessWeights, normalizedReinforcement, reinforcementNumber, 0.5, 20);
	}
	int max = neuron->timesUsedSinceReinforcement;
	float movingAverage = neuron->localLearningRate;
	float correction = (reinforcement > 0) ? 0 : 1;
	//for (int i = 0; i < max; i++) {
		movingAverage = (MOVING_AVERAGE_WEIGHT * correction) + ((1 - MOVING_AVERAGE_WEIGHT) * movingAverage);
	//}
	neuron->localLearningRate = movingAverage;
	neuron->timesUsedSinceReinforcement = 0;
	neuron->timesReinforced++;
}


// consider whether or not the learning rate should be included in the normalizedReinforcment
void applyADAM(float weights[4][WEIGHT_LENGTH], float normalizedReinforcement, unsigned long timesReinforced, float minClamp, float maxClamp) {
	float gradientCorrection = 1 / (1 - powf(GRADIENT_PERSISTENCE, timesReinforced));
	float squareCorrection = 1 / (1 - powf(SQUARE_PERSISTENCE, timesReinforced));
	for (int i = 0; i < WEIGHT_LENGTH; i++) {
		float scaledGradient = weights[1][i] * normalizedReinforcement;
		float gradientAverage = ((1 - GRADIENT_PERSISTENCE) * scaledGradient) + (GRADIENT_PERSISTENCE * weights[2][i]);
		float square = normalizedReinforcement * normalizedReinforcement;
		float squaredGradientAverage = ((1 - SQUARE_PERSISTENCE) * square) + (SQUARE_PERSISTENCE * weights[3][i]);
		float correctedGradientAverage = gradientAverage * gradientCorrection;
		float correctedSquareAverage = squaredGradientAverage * squareCorrection;
		
		float newWeightProposal = weights[0][i] + (INTERNAL_LEARNING_RATE * correctedGradientAverage / (sqrtf(correctedSquareAverage) + 1E-8));
		weights[0][i] = clamp(newWeightProposal, minClamp, maxClamp);
		//printf("%f %f\n", correctedGradientAverage, correctedSquareAverage);
		//assert((weights[0][i]));
		weights[1][i] = 0;
		weights[2][i] = gradientAverage;
		weights[3][i] = squaredGradientAverage;
	}
}

void reinforce(Network* network, float feedback) {
	if (network->numberOfReinforcements == network->allocatedReinforcementHistoryLength) {
		int maxReinforcements = network->networkSize;
		for (int i = 0; i < maxReinforcements; i++) {
			Neuron* neuron = &(network->neurons[i]);
			if (network->mostRecentReinforcement > neuron->lastUpdated && neuron->timesUsedSinceReinforcement > 0) {
				float reinforcement = getReinforcement(network, neuron->lastUpdated);
				updateWeights(neuron, reinforcement);
			}
		}
		network->numberOfReinforcements = 0;
	}
	//printf("%i %i %i\n", (int) numberOfElements, (int) lengthof(feedbackHistory), (int) reinforcementHistoryLength);
	unsigned long currentIteration = network->currentIteration;
	float adjustedFeedback = feedback;//(feedback > 0) ? feedback / log(2 + network->currentIteration - network->mostRecentReinforcement) : feedback; 
	//printf("%f\n", adjustedFeedback);
	assert(network->numberOfReinforcements < network->allocatedReinforcementHistoryLength);
	network->feedbackHistory[network->numberOfReinforcements].iterationNumber = network->currentIteration;
	network->feedbackHistory[network->numberOfReinforcements].feedback = adjustedFeedback;
	network->numberOfReinforcements++;
	assert(network->currentIteration > network->mostRecentReinforcement);
	network->mostRecentReinforcement = currentIteration;
}

float getReinforcement(Network* network, unsigned int lastIterationNumber) {
	int searchIndex = network->numberOfReinforcements - 1;
	if (searchIndex > 0) assert(network->feedbackHistory[searchIndex].iterationNumber > network->feedbackHistory[searchIndex - 1].iterationNumber);
	while (lastIterationNumber > network->feedbackHistory[searchIndex].iterationNumber) {
		searchIndex--;
		assert(searchIndex > 0);
		//assert(searchIndex < lengthof(feedbackHistory));
	}
	assert(network->feedbackHistory[searchIndex].iterationNumber > lastIterationNumber);
	return network->feedbackHistory[searchIndex].feedback;
}

void deleteNetwork(Network* network) {
	free(network->neurons);
	free(network->feedbackHistory);
	free(network);
}


float max(float a, float b) {
	return (a > b) ? a : b;
}

float min(float a, float b) {
	return (a < b) ? a : b;
}


float clamp(float value, float minValue, float maxValue) {
	float result = min(max(value, minValue), maxValue);
	assert(result >= minValue);
	assert(result <= maxValue);
	return result;
}

void resetAccumulations(Network* network) {
	int max = network->networkSize;
	for (int i = 0; i < max; i++) network->neurons[i].cumulativeValue = 0;
}


int getInputLength() {
	return NUMBER_OF_INPUTS;
}

void setInputs(Network* network, int neuronNumber, int* inputIndices) {
	int* neuronInputIndices = network->neurons[neuronNumber].externalInputIndices;
	memcpy(neuronInputIndices, inputIndices, NUMBER_OF_INPUTS * sizeof(int));
	assert(inputIndices[NUMBER_OF_INPUTS - 1] == neuronInputIndices[NUMBER_OF_INPUTS - 1]);
}

void resetNetwork(Network* network) {
	resetAccumulations(network);
	
	if (network->currentIteration != network->mostRecentReinforcement) reinforce(network, 0); // ignores dangling actions since last feedback
	network->currentNeuron = &(network->neurons[0]);
}


int getIterationsSinceReinforcement(Network* network) {
	return (int) (network->currentIteration - network->mostRecentReinforcement);
}

float getNeuronErrorRate(Network* network, int neuron) {
	assert(neuron > -1 && neuron < network->networkSize);
	return 1 - network->neurons[neuron].localLearningRate;
}

float getNeuronAccumulation(Network* network, int neuron) {
	assert(neuron > -1 && neuron < network->networkSize);
	return network->neurons[neuron].cumulativeValue;
}

double trainToConvergence(Network* network, float errorCutoff, float learningRate) {
	setAction(network, 0, 1);
	setAction(network, network->networkSize / 2, 2);
	bool stillTraining = true;
	int iterationsPerStep = 1000;
	float inputs[4][2] = {{1, 1}, {0, 0}, {1, 0}, {0, 1}};
	int outputs[4] = {1, 1, 2, 2};
	float error = 0;
	while (stillTraining && network->currentIteration < 1E7) {
		int errors = 0;
		for (int i = 0; i < iterationsPerStep; i++) {
			//printf("%i\n", (int) network->currentIteration);
			int random = rand() % 4;
			int action = runUntilOutput(network, inputs[random]);
			resetAccumulations(network);
			if (action == outputs[random]) {
				reinforce(network, learningRate);
			} else {
				errors++;
				reinforce(network, -1.0 * learningRate);
			}
		}
		error = ((float) errors) / ((float) iterationsPerStep);
		stillTraining = error > errorCutoff;
	}
	return error;//1 / (((double) error) * ((double) network->currentIteration));
}



