//#define NDEBUG true

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <float.h>
#include <assert.h>
#include <string.h>



#define LEARNING_RATE 0.01
//#define 

#define NUMBER_OF_CONNECTIONS (4 + 1)
#define NUMBER_OF_INPUTS 2//3
#define WEIGHT_LENGTH (NUMBER_OF_CONNECTIONS + NUMBER_OF_INPUTS)
#define COOLING_CONSTANT 1.0
#define NETWORK_NONLOCALITY 2

typedef struct Synapse {
	// 0 are actual weights, 1 is gradients, 2 is gradient average, 3 is squared gradient avg
	float offsetWeights[4][WEIGHT_LENGTH];
	float periodWeights[4][WEIGHT_LENGTH];
	float tightnessWeights[4][WEIGHT_LENGTH];
	float amplitudeWeight[4];
	float localLearningRate;
} Synapse;

typedef struct Neuron {
	unsigned long lastUpdated;
	unsigned int timesUsedSinceReinforcement;
	unsigned long timesReinforced;
	float cumulativeValue; // tracks frequency of use
	int externalInputIndices[NUMBER_OF_INPUTS];
	int connectionIndices[NUMBER_OF_CONNECTIONS];
	Synapse synapses[NUMBER_OF_CONNECTIONS];
	int actionCode;
	float localLearningRate;
} Neuron;

typedef struct Reinforcement {
	unsigned long iterationNumber;
	float feedback;
} Reinforcement;

typedef struct Network {
	int networkSize;
	float neuronAccumulationRate;
	Neuron* neurons;
	Neuron* currentNeuron;
	unsigned long currentIteration;
	unsigned long mostRecentReinforcement;
	int allocatedReinforcementHistoryLength;
	int numberOfReinforcements;
	Reinforcement* feedbackHistory;

} Network;


Network* constructNetwork(int numberOfNeurons, int maxReinforcementHistory, float neuronAccumulationRate);
int runUntilOutput(Network* network, float input[NUMBER_OF_INPUTS]);
void initialize();
int wrapIndex(int index, int size);
float cool(Neuron* neuron, unsigned long currentIteration);
void getInputs(Network* network, float *rawInputs, Neuron *neuron, float *combinedInputVector);
int stepNetwork(Network* network, float rawInputs[NUMBER_OF_INPUTS]);
void setAction(Network* network, int index, int actionCode);
void updateWeights(Neuron *neuron, float reinforcement);
void reinforce(Network* network, float feedback);
float getReinforcement(Network* network, unsigned int lastIterationNumber);
void deleteNetwork(Network* network);
int processNeuron(Neuron* neuron, float* inputs);
float max(float a, float b);
float min(float a, float b);
float clamp(float value, float min, float max);
void resetAccumulations(Network* network);
double trainToConvergence(Network* network, float errorCutoff, float learningRate);
int getInputLength();
void setInputs(Network* network, int neuronNumber, int* inputIndices);
void resetNetwork(Network* network);
int getIterationsSinceReinforcement(Network* network);
float getNeuronErrorRate(Network* network, int neuron);
float getNeuronAccumulation(Network* network, int neuron);
void applyADAM(float weights[4][WEIGHT_LENGTH], float normalizedReinforcement, unsigned long timesReinforced, float minClamp, float maxClamp);


#define lengthof(array) (sizeof(array) / sizeof(array[0]))




