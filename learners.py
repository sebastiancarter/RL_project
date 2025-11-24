# CS 440 Reinforcement Learning Assignment
# (C) 2017-2025 Travis Mandel

#These imports should be sufficient
import random
import math
import environments as envs
import distributions as dists
import mdp
import scipy.optimize

# Interface (abstract) for reinforcement learning agents
# Don't change this
class RLAgent(object):

    # Re-initializes the agent with a given RLEnvironment
    # Returns True if initialization was successful, False otherwise
    def initWithEnvironment(self,env):
        pass

    #Chooses a a valid action based on all the feedback processed so far
    # Actions are represented as integers, so it just needs to return a
    # number between 0 and the number of actions in the environment
    def chooseAction(self, state, t):
        pass

    #Processes an episode, which is a list of (state, action, outcome) tuples.
    def processEpisode(self, episode):
        pass
	
    #Signals to the agent that a new episode is starting up 
    def episodeStart(self):
        pass

    # Returns a number of maximum supported actions, or None if there is no limit
    def maxSupportedActions(self):
        return None


#Implements the Simple Greedy Learner as indicated by the instructions
class SimpleGreedyLearner(RLAgent):

    def __init__(self):
        #YOUR CODE HERE
        pass #remove and return True once implemented
        
    def initWithEnvironment(self,env):
        self.env = env
        self.rewardDict = {} # map from (state, action) to (totalReward, rewardCount)
        # init to zero
        for state in env.getAllStates():
            for action in range(self.env.getNumActions()):
                # skip invalid actions
                if not self.env.isActionValid(state, action):
                    continue
                totalReward = 0
                rewardCount = 0
                self.rewardDict[(state, action)] = (totalReward, rewardCount)
        return True
            
    def chooseAction(self, state, t):
        bestAction = None
        bestMeanReward = None
        for action in range(self.env.getNumActions()):
            if not self.env.isActionValid(state, action):
                continue
            (totalReward, rewardCount) = self.rewardDict[(state, action)]
            if rewardCount == 0:
                return action
            meanReward = totalReward / rewardCount
            if bestMeanReward is None or meanReward > bestMeanReward:
                bestMeanReward = meanReward
                bestAction = action
        return bestAction
            
        # all actions have been tried at least once
            

            
        
    def processEpisode(self, episode):
        for timestep in range(len(episode)):
            (state, action, outcome) = episode[timestep]
            reward = self.env.getReward(state, action, outcome)
            (totalReward, rewardCount) = self.rewardDict[(state, action)]
            totalReward += reward
            rewardCount += 1
            self.rewardDict[(state, action)] = (totalReward, rewardCount)


    def episodeStart(self):
        pass # is this right? yes

    # No need to change this one
    def maxSupportedActions(self):
        return None
		
# Implements the PSRL Algorithm as discussed in class
# Implement the version with alpha_i=1 (uniform)
class PSRL(RLAgent):
    def __init__(self):
        # Prior parameter alpha_i = 1 (uniform) for each outcome
        self.alpha_i = 1
        self.alphas = None  # (s,a,o) -> count
        self.policy = None
        
    def initWithEnvironment(self,env):
        self.env = env
        self.alphas = {}
        # Initialize Dirichlet counts with prior alpha_i for every valid (s,a,o)
        for state in self.env.getAllStates():
            for action in range(self.env.getNumActions()):
                if not self.env.isActionValid(state, action):
                    continue

                self.alphas[(state, action)] = []
                for outcome in range(self.env.getNumOutcomes()):
                    self.alphas[(state, action)].append(self.alpha_i)
        self.policy = None
        return True
        
    def chooseAction(self, state, t):
        return self.policy[(state, t)]
        
    def processEpisode(self, episode):
        # Update posterior counts with observed outcomes
        for (state, action, outcome) in episode:
            # Skip invalid (shouldn't happen if episode well-formed)
            if not self.env.isActionValid(state, action):
                continue
            self.alphas[(state, action)][outcome] += 1
        
            
    def episodeStart(self):
        outcomeProbs = {}
        for state in self.env.getAllStates():
            for action in range(self.env.getNumActions()):
                if not self.env.isActionValid(state, action):
                    continue
                dirichletPrior = dists.DirichletDistribution(self.alphas[(state, action)])
                outcomeProbs[(state,action)] = dirichletPrior.sample()
        self.policy = mdp.MDP(self.env, outcomeProbs).computeOptimalPolicy()

    # No need to change this one
    def maxSupportedActions(self):
        return None




#implements the simplified PPO algorithm as discussed in class
class PPOLearner(RLAgent):
    def __init__(self, priorParam, numBetweenUpdates, epsilon):
        self.thetaCurr = priorParam
        self.numEpsBetweenUpdates = numBetweenUpdates
        self.epsilon = epsilon
        self.episodesBetweenUpdatesList = []
        self.nextCriticDict = {}
        self.criticDict = {}
        self.rewardsToGoPerEpisodeList = []

    def initWithEnvironment(self, env):
        self.env = env
        for state in self.env.getAllStates():
            for timestep in range(self.env.getTimeHorizon()):
                self.criticDict[(state, timestep)] = (0,0) # (count, summedRewards)
                self.nextCriticDict[(state, timestep)] = (0,0) # (count, summedRewards)
        return True

    def chooseAction(self, state, t):
        # choose action based on current policy parameter thetaCurr
        # this is essentially a Bernoulli with parameter thetaCurr
        # and is equivalent to flipping a coin with probability thetaCurr
        if self.thetaCurr > random.random():
            return 1 # choose action 1
        else:
            return 0 # choose action 0

    def processEpisode(self, episode):
        self.episodesBetweenUpdatesList.append(episode)
        rewardToGo = 0
        episodeRewardToGoList = []
        for timestep in range(len(episode)-1, -1, -1):
            (state, action, outcome) = episode[timestep]
            reward = self.env.getReward(state, action, outcome)
            rewardToGo += reward
            episodeRewardToGoList.append(rewardToGo)

            # doing this the long way because I dont like .get            
            if (state, timestep) not in self.nextCriticDict:
                self.nextCriticDict[(state, timestep)] = (0,0) # (count, summedRewards)


            (count, summedrewards) = self.nextCriticDict[(state, timestep)]
            count += 1
            summedrewards += rewardToGo
            self.nextCriticDict[(state, timestep)] = (count, summedrewards) 
        # reverse the list to get correct order
        episodeRewardToGoList.reverse() # .reverse() reverses the list in place
        self.rewardsToGoPerEpisodeList.append(episodeRewardToGoList)
        if len(self.episodesBetweenUpdatesList) >= self.numEpsBetweenUpdates:
            self.updatePolicy()
            self.episodesBetweenUpdatesList = []
            self.resetCriticAndRewardsToGo()


    def objectiveFunc(self, optimizeQuantity, additionalArgs):
        optimizeQuantity = optimizeQuantity[0]
        scaledAdvantageSum = 0 
        (oldTheta, epsilon, critic, rewardsToGoPerEpisodeList, episodesBetweenUpdatesList, clipFunc) = additionalArgs
        for episodeNum in range(len(episodesBetweenUpdatesList)):
            episode = episodesBetweenUpdatesList[episodeNum]
            currRewardToGoList = rewardsToGoPerEpisodeList[episodeNum]
            for timestep in range(len(episode)):
                (state, action, outcome) = episode[timestep]
                (criticCount, criticSum) = critic[(state, timestep)]
                rewardToGo = currRewardToGoList[timestep]
                if criticCount == 0:
                    currAdvantage = rewardToGo # if no critic info, critic guesses 0
                    # this way we avoid division by zero
                else:
                    currAdvantage = rewardToGo - criticSum/criticCount

                if action == 1:
                    pi_old = oldTheta
                    pi_new = optimizeQuantity
                else:
                    pi_old = 1 - oldTheta
                    pi_new = 1 - optimizeQuantity
        
                scaledAdvantageSum += min(currAdvantage * (pi_new / pi_old),
                                currAdvantage * clipFunc(pi_new / pi_old, 
                                                           1 - epsilon, 
                                                           1 + epsilon
                                                        )
                            )
        return -scaledAdvantageSum                
    
    def updatePolicy(self):
        newTheta = scipy.optimize.minimize(self.objectiveFunc, x0=self.thetaCurr, args=((self.thetaCurr, 
                                                                                         self.epsilon, 
                                                                                         self.criticDict.copy(), 
                                                                                         self.rewardsToGoPerEpisodeList.copy(), 
                                                                                         self.episodesBetweenUpdatesList.copy(),
                                                                                         self.clipFunc),), bounds=[(0,1)])
        argminTheta = newTheta.x[0]
        print("Updated theta from " + str(self.thetaCurr) + " to " + str(argminTheta))
        self.thetaCurr = argminTheta
    
    def clipFunc(self, value, lower, upper):
        if value < lower:
            return lower
        elif value > upper:
            return upper
        else:
            return value

    def resetCriticAndRewardsToGo(self):
        self.criticDict = self.nextCriticDict.copy()
        self.nextCriticDict = {}
        for state in self.env.getAllStates():
            for timestep in range(self.env.getTimeHorizon()):
                self.nextCriticDict[(state, timestep)] = (0,0) # (count, summedRewards)
        self.rewardsToGoPerEpisodeList = []

    def episodeStart(self):
        pass
    # No need to change this one
    def maxSupportedActions(self):
        return 2  # Your implementation will only support two actions
    
# Implements the finite horizon Q-Learning algorithm as discussed in class
class QLearner(RLAgent):
    def __init__(self, epsilon, alpha):
        self.alpha = alpha
        self.epsilon = epsilon

    def initWithEnvironment(self,env):
        self.env = env
        self.Q = {} # (state, action, t) -> Q-value
        for state in self.env.getAllStates():
            for action in range(self.env.getNumActions()):
                for timestep in range(self.env.getTimeHorizon()):
                    if not self.env.isActionValid(state, action):
                        continue
                    self.Q[(state, action, timestep)] = 0.0
        return True
     
    def chooseAction(self, state, t):
        if random.random() < self.epsilon:
            # explore, choose random valid action
            validActions = []
            for action in range(self.env.getNumActions()):
                if self.env.isActionValid(state, action):
                    validActions.append(action)
            return random.choice(validActions)
        else:
            # exploit, choose best known action
            bestAction = None
            bestQValue = None
            for action in range(self.env.getNumActions()):
                if not self.env.isActionValid(state, action):
                    continue
                qValue = self.Q[(state, action, t)]
                if bestQValue is None or qValue > bestQValue:
                    bestQValue = qValue
                    bestAction = action
            return bestAction
        
    def processEpisode(self, episode):
        for timestep in range(len(episode)):
            (state, action, outcome) = episode[timestep]
            reward = self.env.getReward(state, action, outcome)
            nextState = self.env.getNextState(state, action, outcome)
            currentQ = self.Q[(state, action, timestep)]
            futureValue = 0.0
            if nextState is not None and timestep < self.env.getTimeHorizon() - 1:
                # find max_a' Q(nextState, a', timestep+1)
                bestNextQ = None
                for nextAction in range(self.env.getNumActions()):
                    if not self.env.isActionValid(nextState, nextAction):
                        continue
                    nextQ = self.Q[(nextState, nextAction, timestep + 1)]
                    if bestNextQ is None or nextQ > bestNextQ:
                        bestNextQ = nextQ
                futureValue = bestNextQ
            # Q-learning update
            newQ = currentQ + self.alpha * (reward + futureValue - currentQ)
            # 
            for timeStep in range(self.env.getTimeHorizon()):
                self.Q[(state, action, timeStep)] = newQ
        
            
    def episodeStart(self):
        pass

    def maxSupportedActions(self):
        return None




