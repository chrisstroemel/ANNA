import ANNA
import gym
import math
import matplotlib.pyplot as plotter

environment = gym.make('CartPole-v0')

LEARNING_RATE = 0.03

numberOfNeurons = 16
network = ANNA.Network(numberOfNeurons, 4, loggingEnabled=True)
for i in range(numberOfNeurons):
	network.setInputs(i, [0, 1, 2, 3])
network.setAction(numberOfNeurons / 4, 0)
network.setAction(numberOfNeurons * 3 / 4, 1)

epochLengthHistory = []
rewardHistory = []
reinforcementHistory = []
movingAverageHistory = []

rewardMovingAverage = 0
for epoch in range(20000):
	observation = environment.reset()
	totalReward = 0
	for step in range(1000):
		#environment.render()
		action = environment.action_space.sample()
		network.runUntilOutput(observation)
		observation, reward, done, info = environment.step(action)
		totalReward += reward
		if done:
			logReward = math.log(totalReward)
			if epoch == 0: rewardMovingAverage = logReward
			difference = logReward - (0.75 * rewardMovingAverage)
			reinforcement = LEARNING_RATE * difference
			#reinforcement = math.copysign(LEARNING_RATE * math.sqrt(math.fabs(difference)), difference)
			#print(totalReward, reinforcement)
			network.reinforce(reinforcement)
			if (logReward > rewardMovingAverage): rewardMovingAverage = (0.03 * logReward) + (0.97 * rewardMovingAverage)

			rewardHistory.append(logReward)
			movingAverageHistory.append(rewardMovingAverage)


			network.resetNetwork()
			break


iterationNumber = range(len(rewardHistory))
#plotter.scatter(iterationNumber, rewardHistory, s=0.2)
#plotter.plot(iterationNumber, movingAverageHistory)
#plotter.show()

network.showPlots(rewardHistory)












