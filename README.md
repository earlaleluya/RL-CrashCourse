# RL-CrashCourse
A Crash Course on Reinforcement Learning, with demonstration on simple cart pole balancing simulation.

## Training
```python demo.py --mode train --algorithm Q --n_episodes 2000 --n_steps 2500 --n_states 256 --save_path examples/python/Q.csv```

## Testing
```python demo.py --mode test --algorithm Q --data_path examples/python/Q.csv```