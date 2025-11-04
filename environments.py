# CS 440 Reinforcement Learning Assignment
# (C) 2017 Travis Mandel

import random
import distributions as dists
import sys
import mdp

#Interface provided to the learner for getting the known aspects of the environment
#Takes in a fully defined environment and "hides" unknown info
#See RLEnvironment (below) for explanations of the methods.
class RLEnvironmentInfo:
    def  __init__(self, env):
        self.__baseenv = env # There's no way to make this truly private in Python,
                            # but do not access it directly or you will fail the
                            # assignment

    def getInitialState(self):
        return self.__baseenv.getInitialState()
        
    def isActionValid(self, state, action):
        return self.__baseenv.isActionValid(state, action)

    def getNextState(self, state, action, outcome):
        return self.__baseenv.getNextState(state,action,outcome)

    def getReward(self, state, action, outcome):
        return self.__baseenv.getReward(state,action,outcome)        

    def getTimeHorizon(self):
        return self.__baseenv.getTimeHorizon()
        
    def getAllStates(self):
        return self.__baseenv.getAllStates()

    def getNumOutcomes(self):
        return self.__baseenv.getNumOutcomes()
    
    def getNumActions(self):
        return self.__baseenv.getNumActions()
		
    def getMinReward(self):
        return self.__baseenv.getMinReward()
		
    def getMaxReward(self):
        return self.__baseenv.getMaxReward()
    

#A fully-defined Reinforcement Learning Environment
class RLEnvironment:

    #Re-initializes the environment from scratch
    #Returns True if reset successful, False otherwise
    def reset(self):
        pass

    #All RL problems for this assignment have a single start state.
    # This function returns it.
    def getInitialState(self):
        pass

    #Returns true if the action is valid at this state, false otherwise
    def isActionValid(self, state, action):
        pass

    # Gets the next state, after starting at <state>,
    # taking <action>, and observing <outcome>.
    # returns None if we terminate instead of transioning
    def getNextState(self, state, action, outcome):
        pass

    # Gets the next reward, after starting at <state>,
    # taking <action>, and observing <outcome>.
    # Should never return None, we should always get a reward.
    def getReward(self, state, action, outcome):
        pass
        
    # Gets the maximum number of timesteps in an episode
    def getTimeHorizon(self):
        pass

    # Gets all possible states (not guaranteed to be reachable)
    def getAllStates(self):
        pass

    #Gets the number of outcomes
    # Remember outcomes are labelled 0,...,numOutcomes-1
    def getNumOutcomes(self):
        pass

    #Gets the number of actions
    # Remember actions are labelled 0,...,numOutcomes-1
    def getNumActions(self):
        pass

    #Gets the minimum theoretically possible reward.
    # You are guaranteed never to get a reward lower than this
    def getMinReward(self):
        pass

    #Gets the maximum theoretically possible reward.
    # You are guaranteed never to get a reward higher than this
    def getMaxReward(self):
        pass

    # Returns the true dynamics of the environment as an MDP
    #
    # Obviously not possible in the real world but
    # doable for toy problems like those on this assignment.
    def getTrueMDP(self):
        pass

		

# Riverswim Environment: Linear chain of 6 states,
#  with misleading but easy to get reward
class RiverswimEnvironment:
    def  __init__(self):
        self.NUM_STATES = 6
        
        self.LEFT_OUTCOME = 0
        self.RIGHT_OUTCOME = 1
        self.STAY_WITH_0_OUTCOME = 2
        self.STAY_WITH_01_OUTCOME = 3
        self.STAY_WITH_1_OUTCOME = 4
        self.NUM_OUTCOMES = 5

        self.LEFT_ACTION = 0
        self.RIGHT_ACTION = 1

    def reset(self):
        return True

    def getInitialState(self):
        return 0

    def isActionValid(self, state, action):
        self.checkState(state)
        self.checkAction(action)
        return True

    def getNextState(self, state, action, outcome):
        self.checkState(state)
        self.checkAction(action)
        self.checkOutcome(outcome)
        if outcome == self.LEFT_OUTCOME:
            return max(state-1, 0)
        if outcome == self.STAY_WITH_0_OUTCOME or \
           outcome == self.STAY_WITH_01_OUTCOME or \
           outcome == self.STAY_WITH_1_OUTCOME:
            return state
        if outcome == self.RIGHT_OUTCOME:
            return min(state+1, self.NUM_STATES-1)

    def getReward(self, state, action, outcome):
        self.checkState(state)
        self.checkAction(action)
        self.checkOutcome(outcome)
        if outcome == self.STAY_WITH_01_OUTCOME:
            return .01
        elif outcome == self.STAY_WITH_1_OUTCOME:
            return 1
        else:
            return 0
        
    def checkState(self,state):
        if state < 0 or state >= self.NUM_STATES:
            print("Invalid state detected in Riverswim: " + str(state))
            sys.exit(-1)
            
    def checkAction(self,action):
        if action < 0 or action > self.getNumActions():
            print("Invalid action detected in Riverswim: " + str(action))
            sys.exit(-1)

    def checkOutcome(self,outcome):
        if outcome < 0 or outcome > self.getNumOutcomes():
            print("Invalid outcome detected in Riverswim: " + str(outcome))
            sys.exit(-1)        

    def getTimeHorizon(self):
        return 10

    def getAllStates(self):
        return list(range(self.NUM_STATES))

    def getNumOutcomes(self):
        return self.NUM_OUTCOMES

    def getNumActions(self):
        return 2

    def getMinReward(self):
        return 0

    def getMaxReward(self):
        return 1000

    def getTrueMDP(self):
        obsProbs = self.initOProbs()
        obsProbs[(0,self.LEFT_ACTION)][self.STAY_WITH_01_OUTCOME] = 1.0
        obsProbs[(0,self.RIGHT_ACTION)][self.STAY_WITH_0_OUTCOME] = 0.8
        obsProbs[(0,self.RIGHT_ACTION)][self.RIGHT_OUTCOME] = 0.2
        for i in range(1,self.NUM_STATES-1):
            obsProbs[(i,self.RIGHT_ACTION)][self.LEFT_OUTCOME] = 0.2
            obsProbs[(i,self.RIGHT_ACTION)][self.RIGHT_OUTCOME] = 0.8
            obsProbs[(i,self.LEFT_ACTION)][self.LEFT_OUTCOME] = 1.0

        lst= self.NUM_STATES-1
        obsProbs[(lst,self.LEFT_ACTION)][self.LEFT_OUTCOME] = 1.0
        obsProbs[(lst,self.RIGHT_ACTION)][self.STAY_WITH_1_OUTCOME] = 0.8
        obsProbs[(lst,self.RIGHT_ACTION)][self.LEFT_OUTCOME] = 0.2
        return mdp.MDP(self, obsProbs)

    def initOProbs(self):
        oProbs = {}
        for s in self.getAllStates():
            for a in range(self.getNumActions()):
                if not self.isActionValid(s,a):
                    continue
                oProbs[(s,a)]= [0.0] * self.getNumOutcomes()
        return oProbs


class MazeEnvironment(RLEnvironment):
    # In this maze, you know where the goal is located,
    # but have no clue where the walls and pits are,
    # or how the actions map onto movement.
    def  __init__(self):
        self.WIDTH = 5
        self.HEIGHT = 4
        
        self.NORTH_OUTCOME = 0
        self.SOUTH_OUTCOME = 1
        self.EAST_OUTCOME = 2
        self.WEST_OUTCOME = 3
        self.HIT_WALL_OUTCOME = 4
        self.FELL_PIT_OUTCOME = 5
        self.NUM_OUTCOMES = 6

        #Must keep ordering the same as first outcomes
        self.NORTH_ACTION = 0
        self.SOUTH_ACTION = 1
        self.EAST_ACTION = 2
        self.WEST_ACTION = 3
        self.NUM_ACTIONS = 4

    def reset(self):
        return True

    def getInitialState(self):
        return 0,0

    def checkState(self,state):
        x,y = state
        if x < 0 or x >= self.WIDTH or y <0 or y >= self.HEIGHT:
            print("Invalid state detected in Maze: " + str(state))
            sys.exit(-1)
            
    def checkActionSA(self,state,action):
        self.checkAction(action)
        if not self.isActionValid(state,action):
            print("Invalid action detected in Maze: " + str(action) +\
                  " is not valid at state " + str(state))
            sys.exit(-1)

    def checkAction(self,action):
        if action < 0 or action > self.getNumActions():
            print("Invalid action detected in Maze: " + str(action))
            sys.exit(-1)

    def checkOutcome(self,outcome):
        if outcome < 0 or outcome > self.getNumOutcomes():
            print("Invalid outcome detected in Maze: " + str(outcome))
            sys.exit(-1)
        
    #Returns true if the action is valid at this state, false otherwise
    def isActionValid(self, state, action):
        self.checkState(state)
        self.checkAction(action)
        x,y = state
        if action == self.WEST_ACTION and  x <= 0:
            return False
        elif action == self.EAST_ACTION and x >= self.WIDTH -1:
            return False
        elif action == self.NORTH_ACTION and y <= 0:
            return False
        elif action == self.SOUTH_ACTION and y >= self.HEIGHT -1:
            return False

        return True

    def getNextStateInner(self,x,y,outcome):
        if outcome == self.WEST_OUTCOME:
            return (max(x-1, 0),y)
        elif outcome == self.EAST_OUTCOME:
            return (min(x+1, self.WIDTH-1),y)
        elif outcome == self.NORTH_OUTCOME:
            return (x, max(y-1, 0))
        elif outcome == self.SOUTH_OUTCOME:
            return (x, min(y+1, self.HEIGHT-1))
        #otherwise, no movement
        return x,y
        
    def hitsGoal(self, state, outcome):
        if (state == (2,3) and outcome == self.EAST_OUTCOME) or \
           (state == (4,3) and outcome == self.WEST_OUTCOME):
            return True #Can't get there from north due to a wall
        return False

    def getNextState(self, state, action, outcome):
        #TOO SLOW
        #self.checkState(state)
        #self.checkActionSA(state, action)
        #self.checkOutcome(outcome)
        
        x,y = state

        if self.hitsGoal(state,outcome) or outcome == self.FELL_PIT_OUTCOME:
            return None #Terminal outcomes

        return self.getNextStateInner(x,y,outcome)
        
        

        

    # Gets the next reward, after starting at <state>,
    # taking <action>, and observing <outcome>.
    # Should never return None, we should always get a reward.
    def getReward(self, state, action, outcome):
        #TOO SLOW
        #self.checkState(state)
        #self.checkActionSA(state,action)
        #self.checkOutcome(outcome)

        if outcome == self.FELL_PIT_OUTCOME:
            return -20
        if self.hitsGoal(state, outcome):
            return 100

        return -0.5
        

    def getTimeHorizon(self):
        return 13

    def getAllStates(self):
        allStates = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                allStates.append((x,y))
        return allStates

    def getNumOutcomes(self):
        return self.NUM_OUTCOMES

    def getNumActions(self):
        return self.NUM_ACTIONS

    def getMinReward(self):
        return -1

    def getMaxReward(self):
        return 100

    def getTrueMDP(self):
        obsProbs = self.initOProbs()
        walls = [(0,1),(2,1),(3,2)]
        pit = (1,3)
        for s,a in obsProbs:
            (x,y) = s
            for o in [self.NORTH_OUTCOME, self.SOUTH_OUTCOME, \
                            self.EAST_OUTCOME, self.WEST_OUTCOME]:
                oreal = o
                nextState = self.getNextStateInner(x,y,o)
                if nextState == pit:
                    oreal = self.FELL_PIT_OUTCOME
                if nextState in walls:
                    oreal = self.HIT_WALL_OUTCOME
                if o == a:
                    obsProbs[(s,a)][oreal] += 0.7
                else:
                    obsProbs[(s,a)][oreal] += 0.1
                    
        return mdp.MDP(self,obsProbs)
                
                

    def initOProbs(self):
        oProbs = {}
        for s in self.getAllStates():
            for a in range(self.getNumActions()):
                if not self.isActionValid(s,a):
                    continue
                oProbs[(s,a)]= [0.0] * self.getNumOutcomes()
        return oProbs



        
    
    
