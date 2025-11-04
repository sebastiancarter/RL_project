# CS 440 Reinforcement Learning Assignment
# (C) 2017-2025 Travis Mandel

from matplotlib import pyplot as plt
import learners
import environments as envs




#(Learner, name) tuples
learnerTuples = [ (learners.SimpleGreedyLearner(), "Simple Greedy"),(learners.PSRL(), "PSRL"), (learners.QLearner(0.2, 0.1), "Q-Learner"),
                  (learners.PPOLearner(0.5, 5, 0.05), "PPO")]
results = []
#(environment, name, elimit, iterations, delay) tuples);
# delay of 1 means no delay
#Temporarily removing all but one of these is a good way to test just one environment
envTuples = [(envs.RiverswimEnvironment(), "RiverSwim", 100, 30,1), \
            (envs.RiverswimEnvironment(), "RiverDelay", 100, 30,30), \
             (envs.MazeEnvironment(), "Maze", 1000, 15,1)]
for env,envName, elimit,iterations,delay in envTuples:
    erange = list(range(elimit))
    envReady = env.reset()
    if not envReady:
        continue
    for learner,learnerName in learnerTuples:
        #skip if too many actions for learner
        if learner.maxSupportedActions() is not None and env.getNumActions() > learner.maxSupportedActions():
            continue
        failed = False
        totalLearnerResults = [0.0] * len(erange) #init all-zero array
        for it in range(iterations):
            #set up the learner and the environment
            env.reset()
            trueMDP = env.getTrueMDP()
            inited = learner.initWithEnvironment(envs.RLEnvironmentInfo(env))
            if not inited:
                failed = True
                break
            
            cumReward = 0.0
            
            #Main loop, pulling arms, processing rewards, etc.
            toProcess = []
            for epind in erange:
                learner.episodeStart()
                epR = 0
                epFeedback = []
                state = env.getInitialState()
                for t in range(env.getTimeHorizon()):
                    chosenA = learner.chooseAction(state,t)
                    chosenO = trueMDP.sampleOutcome(state, chosenA)
                    epFeedback.append((state,chosenA,chosenO))
                    epR += env.getReward(state, chosenA, chosenO)
                    state = env.getNextState(state, chosenA, chosenO)

                    if state is None:
                        break
                
                toProcess.append(epFeedback)
                     
                cumReward += epR
                totalLearnerResults[epind] += cumReward
                if epind % delay == 0:  #Only one limited form of delay for now
                    for episode in toProcess:
                        learner.processEpisode(episode)
                        toProcess = []
                    
                        
        if failed: # due to unimplemented learner
            continue # With next learner

        #Average the results over the runs and plot
        avgResults = []
        for j in totalLearnerResults:
            avgResults.append(j/float(iterations))
        plt.plot(erange, avgResults, label=learnerName)

    #Format and write out the plot 
    plt.legend(loc='best')
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Reward")
    plt.savefig(envName + ".png")
    plt.clf()
    plt.cla()
    plt.close()
        
        

