import gymnasium as gym
from stable_baselines3 import PPO

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Initialize the PPO model with MLP policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10_000)

# Save the trained model
model.save("ppo_cartpole")

# Load the trained model
model = PPO.load("ppo_cartpole")

# Evaluate the trained agent
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
