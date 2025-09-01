import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


class Environment:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def plot_circle(self, xc, yc, r1, r2, num_points=200):
        """
        Generates circle coordinates (ellipse if r1 != r2).
        xc, yc : center
        r1, r2 : radii along x and y
        """
        theta = np.linspace(0, 2*np.pi, num_points)
        x = xc + r1 * np.cos(theta)
        y = yc + r2 * np.sin(theta)
        return x, y

    def draw_plot(self, x, theta, pole_length=2, episode=0, step=0, epsilon=0):
        self.ax.clear()

        # Ground robot body
        pxg = [x+1, x-1, x-1, x+1, x+1]
        pyg = [0.25, 0.25, 1.25, 1.25, 0.25]

        # Ground Robot wheels
        pxw1, pyw1 = self.plot_circle(x-0.5, 0.125, 0.125, 0.125)
        pxw2, pyw2 = self.plot_circle(x+0.5, 0.125, 0.125, 0.125)

        # Pole
        pxp = [x, x + pole_length * np.sin(theta)]
        pyp = [1.25, 1.25 + pole_length * np.cos(theta)]

        # Plot
        self.ax.plot(pxg, pyg, 'k-')       # ground robot body
        self.ax.plot(pxw1, pyw1, 'k')      # circle 1
        self.ax.plot(pxw2, pyw2, 'k')      # circle 2
        self.ax.plot(pxp, pyp, 'r-')       # pole

        # Display x and theta values
        self.ax.set_title(f"Episode {episode}, Step {step}:\nx = {x:.2f}, theta = {theta:.2f}\n epsilon = {epsilon:.2f}")
        self.ax.axis([-6, 6, 0, 6])
        self.ax.set_aspect('equal', adjustable='box')  # keep proportions
        self.fig.canvas.draw()
        plt.pause(0.001)
        


    def show(self, agent, state, episode=0, step=0, epsilon=0):
        self.draw_plot(x=state.x, theta=state.theta, pole_length=agent.pole_length, episode=episode, step=step, epsilon=epsilon)


if __name__ == '__main__':
    env = Environment()
    env.draw_plot(x=-2, theta=np.pi/6)
    plt.show()