# CS 440 Reinforcement Learning Assignment
# (C) 2017 Travis Mandel

import random
import distributions as dists
import sys

#Interface provided to the learner for getting the known aspects of the environment
#Takes in a fully defined environment and "hides" unknown info
class MDP:
    #obsProbs must be a map from (s,a) --> list of outcome probabilities
    def  __init__(self, env, obsProbs):
        self.env = env
        self.obsProbs = obsProbs
        if not self.validate():
            sys.exit(-1)
		
    def validate(self):
        for pair in self.obsProbs:
            probs = self.obsProbs[pair]
            if abs(1-sum(probs)) > 0.0001:
                print("Trying to construct invalid mdp! (s,a)= " + \
                      str(pair) + " oProbs " + str(probs))
                return False
            
        return True
            

    # Should return a dict mapping (s,t) --> a
    # Where a is the optimal long-term action to take in this MDP
    def computeOptimalPolicy(self):
        policy = {}
        V = {}
        TimeHorizon = self.env.getTimeHorizon()
        allStates = self.env.getAllStates()
        numOutcomes = self.env.getNumOutcomes()
        numActions = self.env.getNumActions()

        # Backward induction
        for t in range(TimeHorizon - 1, -1, -1):
            for state in allStates:
                bestAction = None
                bestValue = None
                for action in range(numActions):
                    if not self.env.isActionValid(state, action):
                        continue
                    probs = self.obsProbs[(state, action)]
                    actionValue = 0.0
                    for outcome in range(numOutcomes):
                        # immediate reward for this (s,a,o)
                        immediateReward = self.env.getReward(state, action, outcome)
                        # transition to next state (may be terminal/None)
                        nextState = self.env.getNextState(state, action, outcome)
                        futureValue = 0.0
                        # Only add future value if there is a next state and time remains
                        if nextState is not None and t < TimeHorizon - 1:
                            futureValue = V.get((nextState, t + 1), 0.0)
                        actionValue += probs[outcome] * (immediateReward + futureValue)
                    if bestValue is None or actionValue > bestValue:
                        bestValue = actionValue
                        bestAction = action
                    elif actionValue == bestValue:
                        # pick action w/ lowest numeric val if tied
                        # I believe this is maybe redundant because we loop through actions
                        # in order, but just to be safe:
                        if action < bestAction:
                            bestValue = actionValue
                            bestAction = action
                            
                # Record optimal action and value for (state, t)
                policy[(state, t)] = bestAction
                if bestValue is None:
                    V[(state, t)] = 0.0
                else:
                    V[(state, t)] = bestValue
        return policy


    def sampleOutcome(self, state, action):
        if not self.env.isActionValid(state, action):
            print("Error: Trying to take invalid action. " + \
                  "At state " + str(state) + " taking action " + str(action))
            sys.exit(-1)
        probs = self.obsProbs[(state,action)]
        choice = random.random()
        sumP = 0
        for i in range(len(probs)):
            sumP += probs[i]
            if choice < sumP:
                return i

        print("Error! Should have sampled a value by now!")
        sys.exit(-1)
            