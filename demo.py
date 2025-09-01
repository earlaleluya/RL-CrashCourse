from examples.python.cartpole_from_scratch.cartpole_simulation import CartPoleSimulation
import matplotlib.pyplot as plt
import argparse




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This serves as a supplementary resource for teaching Reinforcement Learning, as demonstrated by cartpole balancing simulation.")
    parser.add_argument('--mode', required=False, default='train', help="Preferred mode to run.", choices=['train', 'test'])
    parser.add_argument('--algorithm', required=False, default='Q', help="What algorithm to use.", choices=['Q', 'SARSA'])
    parser.add_argument('--n_episodes', required=False, default=2000, help="Number of episodes.", type=int)
    parser.add_argument('--n_steps', required=False, default=2500, help="Number of steps per episode.", type=int)
    parser.add_argument('--n_states', required=False, default=256, help="Number of states, must comply n_bins**4.", type=int)
    parser.add_argument('--save_path', required=False, default=None, help="Path where to save csv file")
    parser.add_argument('--data_path', required=False, default=None, help="Path where to retrieve csv file")
    args = parser.parse_args()

    cartpole_sim = CartPoleSimulation(
        n_episodes=args.n_episodes, 
        n_steps=args.n_steps, 
        n_states=args.n_states, 
        algorithm=args.algorithm, 
        save_path=args.save_path,
        data_path=args.data_path
        )
       

    plt.ion()
    if args.mode == 'train':
        cartpole_sim.train()
    else:
        cartpole_sim.test(load_path=args.data_path)
    plt.ioff()
    plt.show()