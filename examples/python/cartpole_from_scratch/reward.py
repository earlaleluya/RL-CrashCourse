class Reward:

    def compute_reward(self, state_value): # R_{t+1}
        if state_value > 0:   # no fail
            reward = 1.0
        else:   # fail
            reward = 0.0
        return reward