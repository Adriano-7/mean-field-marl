import re
import matplotlib.pyplot as plt

# File path
file_path = "output.txt"

# Lists to store data
rounds = []
rewards = []

# Regular expressions to match lines
round_pattern = re.compile(r"\[\*\] ROUND #(\d+)")
reward_pattern = re.compile(r"\[INFO\] .*'ave_agent_reward': ([\d\-.e]+)")

# Parse the file
with open(file_path, 'r') as file:
    current_round = None
    for line in file:
        # Check for round line
        round_match = round_pattern.search(line)
        if round_match:
            current_round = int(round_match.group(1))
        
        # Check for reward line
        reward_match = reward_pattern.search(line)
        if reward_match and current_round is not None:
            reward = float(reward_match.group(1))
            rounds.append(current_round)
            rewards.append(reward)
            current_round = None  # Reset for the next round

# Plot the reward evolution
plt.figure(figsize=(10, 6))
plt.plot(rounds, rewards, marker='o', linestyle='-', color='b')
plt.title("Evolution of Reward Function by Epochs")
plt.xlabel("Round")
plt.ylabel("Average Agent Reward")
plt.grid(True)
plt.show()
