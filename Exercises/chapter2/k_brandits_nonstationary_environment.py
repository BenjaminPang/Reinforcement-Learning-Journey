import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    Represents a k-armed bandit problem.

    Args:
        k (int): The number of arms (actions).
    """
    def __init__(self, k=10):
        self.k = k
        # Select action values starting at 0.
        self.action_values = np.zeros(self.k)
        # The optimal action is the one with the highest action value.
        self.optimal_action = np.argmax(self.action_values)

    def change(self):
        """
        Update at each step by an incremental value sampled from a normal distribution with mean 0.0 and var 0.01.
        :return:
        """
        increment = np.random.normal(loc=0.0, scale=0.01, size=self.k)
        self.action_values += increment
        self.optimal_action = np.argmax(self.action_values)

    def get_reward(self, action):
        """
        Returns the reward for taking a given action.
        The reward is selected from a normal distribution with mean q*(action) and variance 1.
        """
        return np.random.normal(self.action_values[action], 1)


class Agent:
    """
    Represents an agent learning to solve the bandit problem.

    Args:
        k (int): The number of arms.
        epsilon (float): The probability of choosing a random action (exploration).
    """
    def __init__(self, k=10, epsilon=0.0, step_size=None):
        self.k = k
        self.epsilon = epsilon
        self.action_value_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)
        self.step_size = step_size

    def select_action(self):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.k)
        else:
            # Exploit: choose the action with the highest estimated value
            return np.argmax(self.action_value_estimates)

    def update_estimates(self, action, reward):
        """
        Updates the action-value estimate for the chosen action using the sample-average method.
        """
        self.action_counts[action] += 1
        if self.step_size is None:
            # Q_n+1 = Q_n + (1/n) * (R_n - Q_n)
            self.action_value_estimates[action] += (1 / self.action_counts[action]) * (reward - self.action_value_estimates[action])
        else:
            self.action_value_estimates[action] += self.step_size * (reward - self.action_value_estimates[action])


def run_experiment(k, num_runs, time_steps):
    """
    Runs the full bandit experiment.

    Args:
        k (int): Number of arms.
        num_runs (int): Number of independent runs.
        time_steps (int): Number of steps per run.

    Returns:
        tuple: A tuple containing the average rewards and optimal action percentages for each epsilon.
    """
    ### we compare two methods one with 1/n learning rate and another has constant step-size, alpha = 0.1
    epsilon = 0.1
    avg_rewards = np.zeros((2, time_steps))
    optimal_action_percentage = np.zeros((2, time_steps))

    agents = [Agent(k, epsilon=0.1), Agent(k, epsilon=0.1, step_size=0.1)]
    for i, agent in enumerate(agents):
        print(f"Running experiment for agent_{i}")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"Run {run + 1}/{num_runs}")

            bandit = Bandit(k)
            for step in range(time_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action)
                agent.update_estimates(action, reward)

                avg_rewards[i, step] += reward
                if action == bandit.optimal_action:
                    optimal_action_percentage[i, step] += 1

                bandit.change()

    avg_rewards /= num_runs
    optimal_action_percentage /= num_runs

    return avg_rewards, optimal_action_percentage


if __name__ == '__main__':
    # Experiment parameters from the text
    K_ARMS = 10
    NUM_RUNS = 2000
    TIME_STEPS = 10000

    # Run the experiment
    avg_rewards, optimal_action_percentage = run_experiment(K_ARMS, NUM_RUNS, TIME_STEPS)

    # --- Plotting the results ---

    # Figure 1: Average Reward
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards[0], label='sample average', color='green')
    plt.plot(avg_rewards[1], label='constant step-size', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward on 10-Armed Testbed, TIME_STEPS: {TIME_STEPS}')
    plt.legend()
    plt.grid(True)
    plt.pause(1)

    # Figure 2: % Optimal Action
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_action_percentage[0] * 100, label='sample average', color='green')
    plt.plot(optimal_action_percentage[1] * 100, label='constant step-size', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action')
    plt.title(f'% Optimal Action on 10-Armed Testbed, TIME_STEPS: {TIME_STEPS}')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.show()
