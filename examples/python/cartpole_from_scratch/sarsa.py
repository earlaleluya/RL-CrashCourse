import numpy as np
import pandas as pd


class SARSA:

    def __init__(self, n_states, n_actions, alpha, gamma, data_path=None, save_path=None):
        self.Q = np.zeros((n_states, n_actions)) if data_path is None else self.load_csv(data_path)
        self.alpha = alpha 
        self.gamma = gamma 
        self.save_path = save_path


    def load_csv(self, data_path):
        df = pd.read_csv(data_path, header=None)
        return df.values


    def load(self, load_path=None):
        if load_path is None:
            self.Q = self.load_csv(self.save_path)  # load from previously saved file
        else:
            self.Q = self.load_csv(load_path)       # load new file


    def update(self, state, action, next_state, next_reward, next_action):
        if next_state < 0:    # fail
            target = next_reward  # No future reward if failed
        else:   # no fail
            target = next_reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * target


    def best_action_idx(self, state):
        return np.argmax(self.Q[state])        


    def save(self):
        pd.DataFrame(self.Q).to_csv(self.save_path, index=False, header=False)