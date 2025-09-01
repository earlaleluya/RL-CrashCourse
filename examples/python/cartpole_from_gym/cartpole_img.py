import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN, PPO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from skimage.transform import resize
from stable_baselines3.common.callbacks import BaseCallback
import os


class CustomEnv(gym.Env):
    """
    A custom environment that returns a grayscale image as observation,
    and can also render using the default CartPole-v1 visualization.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, img_size=(64, 64)):
        super().__init__()
        self.img_size = img_size  # (height, width)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(img_size[0], img_size[1], 1), dtype=np.uint8
        )
        # State variables
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.render_mode = render_mode
        # For default visualization
        self.visual_env = gym.make("CartPole-v1", render_mode="human") if render_mode == "human" else None


    def _get_state(self):
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])

    def _set_state(self, state_arr):
        self.x, self.x_dot, self.theta, self.theta_dot = state_arr

    def _draw_cartpole(self):
        fig, ax = plt.subplots(figsize=(2, 2), dpi=self.img_size[0] // 2)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        cart_y = 0.0
        cart_width = 0.4
        cart_height = 0.2
        ax.add_patch(plt.Rectangle((self.x - cart_width/2, cart_y - cart_height/2),
                                   cart_width, cart_height, color='black'))
        pole_len = 1.0
        pole_x = self.x + pole_len * np.sin(self.theta)
        pole_y = cart_y + cart_height/2 + pole_len * np.cos(self.theta)
        ax.plot([self.x, pole_x], [cart_y + cart_height/2, pole_y], color='gray', linewidth=4)
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = plt.imread(buf)
        if img.shape[2] == 4:
            img = img[..., :3]
        img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        img_gray = resize(img_gray, self.img_size, anti_aliasing=True)
        img_gray = (img_gray * 255).astype(np.uint8)
        img_gray = img_gray[..., np.newaxis]
        return img_gray

    def step(self, action):
        force = 1.0 if action == 1 else -1.0
        self.x += 0.05 * force
        self.theta += 0.05 * force
        reward = 1.0 if abs(self.theta) < 0.2 and abs(self.x) < 2.4 else 0.0
        terminated = abs(self.theta) > 0.2 or abs(self.x) > 2.4
        truncated = False
        observation = self._draw_cartpole()
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        observation = self._draw_cartpole()
        info = {}
        return observation, info

    def render(self):
        # Render the grayscale image (agent's observation)
        img = self._draw_cartpole()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.01)
        plt.clf()
        # Also render using the default CartPole-v1 visualization
        if self.visual_env is not None:
            # Ensure visual_env is reset before rendering
            if not hasattr(self.visual_env, "_has_been_reset") or not self.visual_env._has_been_reset:
                self.visual_env.reset()
            # Sync state to visual_env
            self.visual_env.env.state = self._get_state()
            self.visual_env.render()

    def close(self):
        if self.visual_env is not None:
            self.visual_env.close()


# --- Callback for rendering during training ---
class RenderCallback(BaseCallback):
    def __init__(self, render_freq=100, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.envs[0].render()
        return True


if __name__ == '__main__':
    env = CustomEnv(render_mode="human")
    render_callback = RenderCallback(render_freq=1)
    
    model = DQN("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, callback=render_callback)  
    model.save("examples/python/cartpole_from_gym/dqn_cartpole_img_09022025")
    
    model = DQN.load("examples/python/cartpole_from_gym/dqn_cartpole_img_09022025")
    
    obs, info = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
