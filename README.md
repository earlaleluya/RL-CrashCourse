# RL-CrashCourse
A Crash Course on Reinforcement Learning, with demonstration on cart pole balancing simulation. There are 3 variations of the code, which are as follows:


## 1. Written from Scratch and State Space has 1 Dimension
This version was written using only basic libraries (i.g., NumPy) to provide step-by-step explanation of the core concept of Reinforcement Learning. In the cartpole balancing simulation, the state space is $S \subset \mathbb{N}$, and its state representation is $S=\{s_{bin}\}$, where $s_{bin}=x_{bin}\cdot n^3_{bins} + \dot{x}_{bin}\cdot n^2_{bins} + \theta_{bin}\cdot n_{bins} + \dot{\theta}_{bin}$. We use Q-learning and SARSA for this simple example.

You may view `demo.ipynb` or use the following commands below:<br>
<br>For Training: <br>
```python demo.py --mode train --algorithm Q --n_episodes 2000 --n_steps 2500 --n_states 256 --save_path examples/python/cartpole_from_scratch/Q.csv```

<br>For testing<br>
```python demo.py --mode test --algorithm Q --data_path examples/python/cartpole_from_scratch/Q.csv```


## 2. Using Gymnasium Environment, Stable-baseline3 RL algorithms, and State Space has 4 Dimensions
We use [gymnasium](https://gymnasium.farama.org/) and [stable_baseline3](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html) libraries for the second version of the cartpole balancing simulation. The state space is $S \subset \mathbb{R}^4$ since the state representation is $S=\{x, \dot{x}, \theta, \dot{\theta}\}$. The Q-table is impossible for this state space because it will require infinite rows. That is why we use function approximator like DQN, PPO, etc.

You may run the code by:<br>
```python examples/python/cartpole_from_gym/cartpole_sim.py```

## 3. Using Gymnasium Environment, Stable-baseline3 RL algorithms, and State is a Grayscale Image
The last version of the simulation aims to demonstrate the process of integrating computer vision in the workflow. Because sometimes raw visual input is the only available observation. This is done by customizing the state and the environment. The state space is $S \subset \{0,1,\cdots,255\}^{H\times W\times 1}$.

You may run the code by:
```python examples/python/cartpole_from_gym/cartpole_img.py```