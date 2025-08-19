import numpy as np
from examples.python.state import State
from examples.python.q import Q
from examples.python.sarsa import SARSA


class Agent:
    # Define constants
    g = 9.8     # Gravitational acceleration in m/s2
    cart_mass = 1.0     # Mass of the cart is assumed 1 kg
    pole_mass = 0.1     # Mass of the pole is assumed 0.1 kg
    pole_length = 2.0   # Length of the pole in meters
    force = 10.0        # Magnitude of force in N
    tau = 0.02          # Time interval per step


    def __init__(self, 
                 algorithm, 
                 n_states,          # number of disjoint boxes of state space
                 n_actions,         #  number of actions
                 alpha=0.5,         # For Q-learning: learning rate
                 gamma=0.85,        # For Q-learning: discount factor for future reward
                 data_path=None,    # path to where data is to be loaded
                 save_path=None,    # path to where data is to be saved
                 ):
        self.n_states = n_states
        self.n_actions = n_actions
        if algorithm=='Q':
            self.algorithm = Q(n_states, n_actions, alpha, gamma, data_path, save_path)
        elif algorithm=='SARSA':
            self.algorithm = SARSA(n_states, n_actions, alpha, gamma, data_path, save_path)
        else:
            raise ValueError("Invalid algorithm")



    def move(self, action, state):
        # Define variables
        m = self.cart_mass + self.pole_mass
        g = self.g
        theta = state.theta        
        theta_dot = state.theta_dot
        f =  -self.force if action=='left' else self.force
        mp = self.pole_mass
        l = self.pole_length

        # Compute pole angular acceleration
        theta_acc = ((m*g*np.sin(theta)) - (np.cos(theta))*(f + mp*l*theta_dot**2*np.sin(theta))) / (((4.0/3.0)*m*l)-(mp*l*np.cos(theta)**2))
        
        # Compute ground robot acceleration
        x_acc = (1.0/m) * (f + (mp*l)*((theta_dot**2*np.sin(theta))-(theta_acc*np.cos(theta)))) 

        # Update states
        next_x = state.x + (self.tau * state.x_dot)
        next_x_dot = state.x_dot + (self.tau * x_acc)
        next_theta = state.theta + (self.tau * state.theta_dot)
        next_theta_dot = state.theta_dot + (self.tau * theta_acc)

        # Create next state obj
        next_state = State(n_states=state.n_states, noise=state.noise)
        next_state.x = next_x 
        next_state.x_dot = next_x_dot
        next_state.theta = next_theta
        next_state.theta_dot = next_theta_dot
        return next_state
    

    def update(self, state, action, next_state, next_reward, next_action=None):
        if isinstance(self.algorithm, Q):
            self.algorithm.update(state, action, next_state, next_reward)
        elif isinstance(self.algorithm, SARSA):
            self.algorithm.update(state, action, next_state, next_reward, next_action)


    def best_action_idx(self, state):
        return self.algorithm.best_action_idx(state=state.compute_state_value())
        

    def save(self):
        self.algorithm.save()

    
    def load(self, load_path=None):
        self.algorithm.load(load_path)