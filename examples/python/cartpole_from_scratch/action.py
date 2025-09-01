import random
import numpy as np


class Action:
    def __init__(self):
        self.actions = ['left', 'right']
        self.action = None 

    def get_action(self, strategy, agent, state, epsilon=0):
        if strategy=='explore_exploit':
            return self.decide(epsilon, agent, state)
        elif strategy=='exploit':
            return self.exploit(agent, state) 



    def decide(self, epsilon, agent, state):
        if np.random.rand() < epsilon:  # implements epsilon-greedy
            self.action = self.explore()
        else:
            self.action = self.exploit(agent, state)
        return self.action 
    

    def explore(self):
        return random.choice(self.actions)


    def exploit(self, agent, state):
        return self.actions[agent.best_action_idx(state)]
     
    
    def idx(self):
        return self.actions.index(self.action)
       