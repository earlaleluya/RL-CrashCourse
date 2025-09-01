import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, PPO



class CustomEnv(gym.Env):
    """
    A custom environment that follows the Gymnasium API.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}


    def __init__(self, render_mode=None):
        super().__init__()
        
        # 1. Action Space
        # Example: a discrete space with 2 actions (e.g., left, right)
        self.action_space = spaces.Discrete(2)

        # 2. Observation Space
        # Example: a continuous space with a single observation (e.g., position)
        self.observation_space = spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32)
        
        # 3. Environment State
        # You'll need to define the initial state of your environment here
        self.state = 0  # Example: initial position is 0
        
        # 4. Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # If your environment needs a visualizer, initialize it here


    def step(self, action):
        """
        Executes one time step within the environment.
        """
        # 1. Take action
        if action == 0:  # Example: move left
            self.state -= 1
        else:  # Example: move right
            self.state += 1

        # 2. Calculate new state, reward, and termination flags
        # Define the logic for how the environment's state changes based on the action.
        
        # Normalize reward to encourage or discourage certain actions
        reward = -abs(self.state)  # Example: reward is the negative distance from 0
        
        # Define termination and truncation conditions
        terminated = False  # e.g., if you reached a goal state
        truncated = False   # e.g., if you run out of time
        
        # 3. Get the observation and info
        # The observation is what the agent sees
        observation = np.array([self.state], dtype=np.float32)
        info = {}  # Optional: additional debugging info
        
        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)  # Always call this first
        
        # Define the logic to reset your environment's state
        self.state = 0  # Reset to initial position
        
        # Get the initial observation and info
        observation = np.array([self.state], dtype=np.float32)
        info = {}
        
        return observation, info


    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode == "human":
            # Implement your visualization logic here
            print(f"Current state: {self.state}")
            pass  # Replace with actual rendering code

    def close(self):
        """
        Clean up resources (e.g., close windows, files).
        """
        # Close any open windows or files here
        pass




if __name__ == '__main__':
    # Create your custom environment
    env = CustomEnv(render_mode="human")
    
    # Initialize the model 
    model = PPO("MlpPolicy", env, verbose=1)    # replace MlpPolicy to CnnPolicy if state is an image

    # Train the agent
    model.learn(total_timesteps=10_000)

    # Save the trained model
    model.save("ppo_template")

    # Load the trained model
    model = PPO.load("ppo_template")

    # Evaluate the trained agent
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
