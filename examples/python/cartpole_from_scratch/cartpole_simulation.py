from examples.python.cartpole_from_scratch.environment import Environment
from examples.python.cartpole_from_scratch.state import State
from examples.python.cartpole_from_scratch.reward import Reward
from examples.python.cartpole_from_scratch.action import Action
from examples.python.cartpole_from_scratch.agent import Agent


class CartPoleSimulation:

    def __init__(self, 
                 n_episodes,    # number of episodes
                 n_steps,       # number of steps per episode
                 n_states,      # number of states, must be n_bins_per_var**4 since [x, x_dot, theta, theta_dot0]
                 noise=0.01,         # magnitude of noise added to starting state
                 epsilon_0=0.99,     # initial exploration-exploitation ratio
                 algorithm='Q',      # method to be used 
                 alpha=0.5,          # Q-learning: learning rate
                 gamma=0.85,         # Q-learning: discount factor for future reward
                 data_path=None,     # path to where data is to be loaded
                 save_path=None,     # path to where data is to be saved
                 ):        
        self.n_episodes = n_episodes
        self.n_steps = n_steps 
        self.env = Environment()
        self.state = State(n_states=n_states, noise=noise)
        self.reward = Reward()
        self.action = Action()
        self.epsilon = epsilon_0
        self.epsilon_decay = (1.0 / n_episodes)
        self.agent = Agent(algorithm=algorithm, n_states=n_states, n_actions=len(self.action.actions), alpha=alpha, gamma=gamma, data_path=data_path, save_path=save_path)
      


    def train(self):
        # Display initial position
        self.env.show(self.agent, self.state)
        
        # Start loop
        for episode in range(self.n_episodes):
            for step in range(self.n_steps):
                
                # Get current state  
                state = self.state
                
                # Determine an action based on current state  
                action = self.action.get_action(strategy='explore_exploit', agent=self.agent, state=state, epsilon=self.epsilon)

                # Update environment
                next_state = self.agent.move(action, state)
                self.env.show(self.agent, next_state, episode=episode+1, step=step+1, epsilon=self.epsilon)

                # Get reward
                next_reward = self.reward.compute_reward(state_value=next_state.compute_state_value())

                # Update agent's learnability
                self.agent.update(state=state.compute_state_value(), action=self.action.idx(), next_state=next_state.compute_state_value(), next_reward=next_reward)

                # Update state
                self.state.update(new_state=next_state)

                # Reset if agent fails
                if next_state.is_fail():
                    break
            
            # Reset state and update epsilon on each episode
            self.epsilon *= (1.0 - self.epsilon_decay)
            self.state.reset()
            self.agent.save()



    def test(self, load_path=None):
        # Display initial position
        self.state.reset()
        self.env.show(self.agent, self.state)

        # Load the saved progress
        self.agent.load(load_path)

        # Start loop
        step = 1
        while not self.state.is_fail():

            # Get current state
            state = self.state 

            # Determine an action based on current state  
            action = self.action.get_action(strategy='exploit', agent=self.agent, state=state)

            # Update environment
            next_state = self.agent.move(action, state)
            self.env.show(self.agent, next_state, step=step)
            step += 1

            # Update state
            self.state.update(new_state=next_state)
