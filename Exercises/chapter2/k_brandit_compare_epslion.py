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
        # Select action values from a normal distribution with mean 0 and variance 1.
        self.action_values = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        # The optimal action is the one with the highest action value.
        self.optimal_action = np.argmax(self.action_values)

    def change(self):
        increment = np.random.normal(loc=0.0, scale=0.01, size=self.k)
        self.action_values += increment

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
    def __init__(self, k=10, epsilon=0.0):
        self.k = k
        self.epsilon = epsilon
        self.action_value_estimates = np.zeros(k)
        self.action_counts = np.zeros(k)

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
        # Q_n+1 = Q_n + (1/n) * (R_n - Q_n)
        self.action_value_estimates[action] += (1 / self.action_counts[action]) * (reward - self.action_value_estimates[action])


def run_experiment(k, num_runs, time_steps, epsilons):
    """
    Runs the full bandit experiment.

    Args:
        k (int): Number of arms.
        num_runs (int): Number of independent runs.
        time_steps (int): Number of steps per run.
        epsilons (list of float): A list of epsilon values to test.

    Returns:
        tuple: A tuple containing the average rewards and optimal action percentages for each epsilon.
    """
    avg_rewards = np.zeros((len(epsilons), time_steps))
    optimal_action_percentage = np.zeros((len(epsilons), time_steps))

    for i, epsilon in enumerate(epsilons):
        print(f"Running experiment for epsilon = {epsilon}")
        for run in range(num_runs):
            if (run + 1) % 200 == 0:
                print(f"  Run {run + 1}/{num_runs}")

            bandit = Bandit(k)
            agent = Agent(k, epsilon)

            for step in range(time_steps):
                action = agent.select_action()
                reward = bandit.get_reward(action)
                agent.update_estimates(action, reward)

                avg_rewards[i, step] += reward
                if action == bandit.optimal_action:
                    optimal_action_percentage[i, step] += 1

    avg_rewards /= num_runs
    optimal_action_percentage /= num_runs

    return avg_rewards, optimal_action_percentage


if __name__ == '__main__':
    # Experiment parameters from the text
    K_ARMS = 10
    NUM_RUNS = 2000
    TIME_STEPS = 1000
    EPSILONS = [0, 0.01, 0.1]  # Greedy, epsilon=0.01, epsilon=0.1

    # Run the experiment
    avg_rewards, optimal_action_percentage = run_experiment(K_ARMS, NUM_RUNS, TIME_STEPS, EPSILONS)

    # --- Plotting the results ---

    # Figure 1: Average Reward
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards[0], label='ε = 0 (Greedy)', color='green')
    plt.plot(avg_rewards[1], label='ε = 0.01', color='red')
    plt.plot(avg_rewards[2], label='ε = 0.1', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.title('Average Reward on 10-Armed Testbed')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # Figure 2: % Optimal Action
    plt.figure(figsize=(12, 6))
    plt.plot(optimal_action_percentage[0] * 100, label='ε = 0 (Greedy)', color='green')
    plt.plot(optimal_action_percentage[1] * 100, label='ε = 0.01', color='red')
    plt.plot(optimal_action_percentage[2] * 100, label='ε = 0.1', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('% Optimal Action')
    plt.title('% Optimal Action on 10-Armed Testbed')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)
    plt.show()
