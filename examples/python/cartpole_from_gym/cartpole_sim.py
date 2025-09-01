import gymnasium as gym
from stable_baselines3 import DQN, PPO

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Initialize the DQN model with MLP policy
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10_000)

# Save the trained model
model.save("examples/python/cartpole_from_gym/dqn_cartpole")

# Load the trained model
model = DQN.load("examples/python/cartpole_from_gym/dqn_cartpole")

# Evaluate the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
