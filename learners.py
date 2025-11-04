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
        pass # is this right?

    # No need to change this one
    def maxSupportedActions(self):
        return None
		
# Implements the PSRL Algorithm as discussed in class
# Implement the version with alpha_i=1 (uniform)
class PSRL(RLAgent):
    def __init__(self):
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

    # No need to change this one
    def maxSupportedActions(self):
        return None
		









#implements the simplified PPO algorithm as discussed in class
class PPOLearner(RLAgent):
    def __init__(self, priorParam, numBetweenUpdates, epsilon):
        # YOUR CODE HERE
        pass  # remove once implemented

        
    def initWithEnvironment(self, env):
        # YOUR CODE HERE
        pass  # remove and return True once implemented

    def chooseAction(self, state, t):
        # YOUR CODE HERE
        pass  # remove once implemented

    def processEpisode(self, episode):
        # YOUR CODE HERE
        pass  # remove once implemented

    def updatePolicy(self):
        # YOUR CODE HERE
        pass  # remove once implemented

    def episodeStart(self):
        pass # is this right?

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




