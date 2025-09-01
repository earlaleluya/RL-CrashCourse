import numpy as np


class State:
    x = 0       # cart position in meters 
    x_dot = 0   # cart velocity
    theta = 0   # pole angle in radians
    theta_dot = 0 # pole angular velocity per 'tau' second
    
    def __init__(self, n_states, noise=0.01):
        self.n_states = n_states
        self.noise = noise
        self.reset()
        self.bounds = {
            'x': [-5.0, 5.0],
            'x_dot': [-1.0, 1.0],
            'theta': [np.deg2rad(-12), np.deg2rad(12)],
            'theta_dot': [-1.0, 1.0]    # 1 rad/tau = 57.3 deg/tau
        }
        self.n_bins = int(round(self.n_states ** 0.25))  # 4th root of n_states
    

    def reset(self,):
        self.x = np.random.uniform(-self.noise, self.noise)
        self.x_dot = np.random.uniform(-self.noise, self.noise)
        self.theta = np.random.uniform(-self.noise, self.noise)
        self.theta_dot = np.random.uniform(-self.noise, self.noise)
        

    def compute_state_value(self):
        if self.is_fail():
            return -1
        x_bin = self.discretize(self.x, 'x')
        x_dot_bin = self.discretize(self.x_dot, 'x_dot')
        theta_bin = self.discretize(self.theta, 'theta')
        theta_dot_bin = self.discretize(self.theta_dot, 'theta_dot')
        return (x_bin * self.n_bins**3) + (x_dot_bin * self.n_bins**2) + (theta_bin * self.n_bins) + theta_dot_bin
        

    def is_fail(self):
        robot_exceeds_left = (self.x < self.bounds['x'][0])
        robot_exceeds_right = (self.x > self.bounds['x'][1])
        pole_falls_at_left = (self.theta < self.bounds['theta'][0])
        pole_falls_at_right = (self.theta > self.bounds['theta'][1])
        return (robot_exceeds_left or robot_exceeds_right or pole_falls_at_left or pole_falls_at_right)


    def discretize(self, var, var_name):
        n_states_per_var = self.n_bins
        # Define the min and max
        [min_val, max_val] = self.bounds[var_name]
        # Clip var to the range
        var_clipped = np.clip(var, min_val, max_val)
        # Compute the bin index
        bin_width = (max_val - min_val) / n_states_per_var
        bin_idx = int((var_clipped - min_val) / bin_width)
        # Ensure bin_idx is within [0, n_states_per_var-1]
        bin_idx = min(bin_idx, n_states_per_var - 1)
        return bin_idx
    

    def update(self, new_state):
        self.x = new_state.x
        self.x_dot = new_state.x_dot
        self.theta = new_state.theta
        self.theta_dot = new_state.theta_dot
        del new_state



