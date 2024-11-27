import random
import numpy as np
import matplotlib.pyplot as plt

TARGET = 1000
VARIANCE = 80
N_AGENTS = 200
N_EPISODES = 2000
INITIAL_EPSILON = 1
MIN_EPSILON = 0.01
DECAY_RATE = 0.995


def G(x):
    return x * np.e ** (-(x-TARGET)**2 / VARIANCE**2)


class Agent:

    def __init__(self):
        self.action = None
        self.q_values = [(np.inf, 0)] * 10

    def choose_action(self, epsilon):
        if random.random() < epsilon:
            self.action = random.choice(range(10))
        else:
            self.action = np.argmax([q for q, _ in self.q_values])
        return self.action

    def update(self, reward):
        q, n = self.q_values[self.action]
        if n == 0:
            self.q_values[self.action] = (reward, 1)
        else:
            self.q_values[self.action] = (q * n + reward) / (n + 1), n + 1
    
    def best_action(self):
        return self.choose_action(0)
    

class GSExperiment:

    def __init__(self, n_agents, n_episodes):
        self.agents = [Agent() for _ in range(n_agents)]
        self.sums = []
        self.n_episodes = n_episodes
        self.results = []

    def train(self):
        epsilon = INITIAL_EPSILON
        for _ in range(self.n_episodes):
            s = 0
            for agent in self.agents:
                s += agent.choose_action(epsilon)
            reward = G(s)
            self.sums.append(s)
            for agent in self.agents:
                agent.update(reward)
            epsilon = max(MIN_EPSILON, epsilon * DECAY_RATE)

    def get_results(self):
        for agent in self.agents:
            self.results.append(agent.best_action())
        return self.results

    def plot_results(self):
        plt.hist(self.results, bins=range(11))
        plt.title('Distribution of actions')
        plt.axvline(x=TARGET//N_AGENTS, color='r', linestyle='--')
        plt.figure()
        plt.plot(self.sums)
        plt.axhline(y=TARGET, color='r', linestyle='-')
        plt.axhline(y=TARGET + VARIANCE, color='r', linestyle='--')
        plt.axhline(y=TARGET - VARIANCE, color='r', linestyle='--')
        plt.title('Sum of actions per episode')
        plt.show()


def plot_reward():
    x = np.linspace(TARGET - 3 * VARIANCE, TARGET + 3 * VARIANCE, 1000)
    y = G(x)
    plt.plot(x, y)
    plt.title('Reward function')
    plt.show()

def experience():
    experiment = GSExperiment(N_AGENTS, N_EPISODES)
    experiment.train()
    result = experiment.get_results()
    final_reward = G(sum(result))
    performance = round((final_reward / G(TARGET))*100, 2)
    print(f'Obtained: {sum(result)} | Expected: {TARGET} | Reward: {final_reward} | Performance: {performance}')
    experiment.plot_results()

if __name__ == '__main__':
    #plot_reward()
    experience()
