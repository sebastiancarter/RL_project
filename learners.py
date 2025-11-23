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
        self.counts = None  # (s,a,o) -> count
        self.policy = None
        
    def initWithEnvironment(self,env):
        self.env = env
        self.counts = {}
        # Initialize Dirichlet counts with prior alpha_i for every valid (s,a,o)
        for state in self.env.getAllStates():
            for action in range(self.env.getNumActions()):
                if not self.env.isActionValid(state, action):
                    continue
                for outcome in range(self.env.getNumOutcomes()):
                    self.counts[(state, action, outcome)] = self.alpha_i
        self.policy = None
        return True
        
    def chooseAction(self, state, t):
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
            self.counts[(state, action, outcome)] += 1
        
            
    def episodeStart(self):
        # Sample a model from the current Dirichlet posteriors
        obsProbs = {}
        for state in self.env.getAllStates():
            for action in range(self.env.getNumActions()):
                if not self.env.isActionValid(state, action):
                    continue
                # Gather counts for (s,a,·)
                outcomeCounts = [self.counts[(state, action, o)] for o in range(self.env.getNumOutcomes())]
                # Sample Dirichlet by sampling gammas and normalizing
                gammaSamples = []
                for c in outcomeCounts:
                    # shape=c, scale=1 gives Gamma(c,1)
                    gammaSamples.append(random.gammavariate(c, 1.0))
                total = sum(gammaSamples)
                # Avoid division by zero (shouldn't occur because c>=1) but safeguard
                if total == 0:
                    probs = [1.0 / self.env.getNumOutcomes()] * self.env.getNumOutcomes()
                else:
                    probs = [g / total for g in gammaSamples]
                obsProbs[(state, action)] = probs
        # Build MDP with sampled transition outcome probabilities and compute optimal policy
        sampledMDP = mdp.MDP(self.env, obsProbs)
        self.policy = sampledMDP.computeOptimalPolicy()

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




