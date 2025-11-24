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
        # TODO: remove this is none thing, redundant and will lose points
        if self.policy is None:
            # no policy yet, choose random valid action
            validActions = []
            for action in range(self.env.getNumActions()):
                if self.env.isActionValid(state, action):
                    validActions.append(action)
            return random.choice(validActions)
        else:
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
        self.policy = None
        self.episodesBetweenUpdatesList = []
        self.criticDict = {}
        self.estQvalsDict = {}

    def initWithEnvironment(self, env):
        self.env = env
        return True

    def chooseAction(self, state, t):
        # TODO: add a comment explaining why
        if self.thetaCurr > random.random():
            return 1 # choose action 1
        else:
            return 0 # choose action 0

    def processEpisode(self, episode):
        self.episodesBetweenUpdatesList.append(episode)
        rewardToGo = 0
        for timestep in range(len(episode)-1, -1, -1):
            (state, action, outcome) = episode[timestep]
            reward = self.env.getReward(state, action, outcome)
            rewardToGo += reward

            # doing this the long way because I dont like .get            
            if (state, action, timestep) not in self.estQvalsDict:
                self.estQvalsDict[(state, action, timestep)] = (0,0) # (count, summedRewards)
            if (state, timestep) not in self.criticDict:
                self.criticDict[(state, timestep)] = (0,0) # (count, summedRewards)
             
            (count, summedrewards) = self.estQvalsDict[(state, action, timestep)]
            count += 1
            summedrewards += rewardToGo
            self.estQvalsDict[(state, action, timestep)] = (count, summedrewards)

            (count, summedrewards) = self.criticDict[(state, timestep)]
            count += 1
            summedrewards += rewardToGo
            self.criticDict[(state, timestep)] = (count, summedrewards) 
    

    def objectiveFunc(self, optimizeQuantity, additionalArgs):
        thingSum = 0 # TODO: choose better name and check if dividing optimizeQuantity by thetaCurr is correct
        (oldTheta, epsilon, critic, estQvals, clipFunc) = additionalArgs
        for episode in self.episodesBetweenUpdatesList:
            for timestep in range(len(episode)):
                (state, action, outcome) = episode[timestep]
                (criticCount, criticSum) = critic[(state, timestep)]
                (estQCount, estQSum) = estQvals[(state, action, timestep)]
                currAdvantage = estQSum/estQCount - criticSum/criticCount
                thingSum += min(currAdvantage * (optimizeQuantity / oldTheta),
                                currAdvantage * clipFunc(optimizeQuantity / oldTheta, 
                                                           1 - epsilon, 
                                                           1 + epsilon
                                                        )
                            )
        return thingSum                
    
    def updatePolicy(self):
        newTheta = scipy.optimize.minimize(self.objectiveFunc, x0=self.thetaCurr, args=((self.thetaCurr, self.epsilon, self.criticDict, self.estQvalsDict, self.clipFunc),), bounds=[(0,1)])
        argminTheta = newTheta.x[0]
        self.thetaCurr = argminTheta
    
    def clipFunc(self, value, lower, upper):
        if value < lower:
            return lower
        elif value > upper:
            return upper
        else:
            return value

    def resetCriticAndEstQvals(self):
        self.criticDict = {}
        self.estQvalsDict = {}
        for action in range(self.env.getNumActions()):
            for state in self.env.getAllStates():
                for timestep in range(self.env.getTimeHorizon()):
                    self.criticDict[(state, timestep)] = (0,0) # (count, summedRewards)
                    self.estQvalsDict[(state, action, timestep)] = (0,0) # (count, summedRewards)

    def episodeStart(self):
        if len(self.episodesBetweenUpdatesList) - 1 == self.numEpsBetweenUpdates:
            self.updatePolicy()
            self.episodesBetweenUpdatesList = []
            self.resetCriticAndEstQvals()

    # No need to change this one
    def maxSupportedActions(self):
        return 2  # Your implementation will only support two actions
    
# Implements the finite horizon Q-Learning algorithm as discussed in class
class QLearner(RLAgent):
    def __init__(self, epsilon, alpha):
        #YOUR CODE HERE
        pass #remove once implemented
        
    def initWithEnvironment(self,env):
        #YOUR CODE HERE
        pass #remove and return True once implemented
        
    def chooseAction(self, state, t):
        #YOUR CODE HERE
        pass #remove once implemented
        
    def processEpisode(self, episode):
        #YOUR CODE HERE
        pass #remove once implemented
            
    def episodeStart(self):
        pass # is this right?

    def maxSupportedActions(self):
        return None




