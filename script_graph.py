import re
import matplotlib.pyplot as plt

# File paths
file_path1 = "mfaclogs.txt"
file_path2 = "mfqlogs.txt"

# Regular expressions to match lines
round_pattern = re.compile(r"\[\*\] ROUND #(\d+)")
reward_pattern = re.compile(r"\[INFO\] .*'total_reward': ([\d\-.e]+)")

def parse_file(file_path):
    """Parse the log file and extract rounds and rewards."""
    rounds = []
    rewards = []
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
    return rounds, rewards

# Parse both files
rounds1, rewards1 = parse_file(file_path1)
rounds2, rewards2 = parse_file(file_path2)

# Plot the reward evolution for both files
plt.figure(figsize=(10, 6))

# Plot data from the first file
plt.plot(rounds1, rewards1, marker='o', linestyle='-', color='b', label="MFAC Logs")

# Plot data from the second file
plt.plot(rounds2, rewards2, marker='o', linestyle='-', color='r', label="MFQ Logs")

# Add titles and labels
plt.title("Evolution of Reward Function by Epochs")
plt.xlabel("Round")
plt.ylabel("Average Agent Reward")
plt.legend()  # Add legend to distinguish between the two files
plt.grid(True)

# Show the plot
plt.show()
