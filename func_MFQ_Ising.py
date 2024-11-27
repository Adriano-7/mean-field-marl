import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import examples.ising_model as ising_model
from examples.ising_model.multiagent.environment import IsingMultiAgentEnv


def parse_arguments():
    """
    Parse command-line arguments for the Ising Model simulation.
    """
    parser = argparse.ArgumentParser(description="Ising Model Multi-Agent Reinforcement Learning Simulation")
    parser.add_argument('-n', '--num_agents', default=100, type=int)
    parser.add_argument('-t', '--temperature', default=1, type=float)
    parser.add_argument('-epi', '--episode', default=1, type=int)
    parser.add_argument('-ts', '--time_steps', default=10000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    parser.add_argument('-dr', '--decay_rate', default=0.99, type=float)
    parser.add_argument('-dg', '--decay_gap', default=2000, type=int)
    parser.add_argument('-ac', '--act_rate', default=1.0, type=float)
    parser.add_argument('-ns', '--neighbor_size', default=4, type=int)
    parser.add_argument('-s', '--scenario', default='Ising.py', help='Path of the scenario Python script.')
    parser.add_argument('-p', '--plot', default=1, type=int)    
    return parser.parse_args()


def setup_environment(args, scenario):
    """
    Set up the multi-agent environment for the Ising model.
    """

    env = IsingMultiAgentEnv(world=scenario.make_world(num_agents=args.num_agents,
                                                      agent_view=1),
                         reset_callback=scenario.reset_world,
                         reward_callback=scenario.reward,
                         observation_callback=scenario.observation,
                         done_callback=scenario.done)
    
    return (
        env, 
        env.n,  # number of agents
        env.observation_space[0].n,  # number of states
        env.action_space[0].n,  # number of actions
        args.neighbor_size + 1  # dimension of Q-state
    )


def boltzmann_exploration(Q, temperature, state, agent_index, n_actions):
    """
    Perform Boltzmann exploration to select an action.
    """
    action_probs_numes = []
    denom = 0
    
    for i in range(n_actions):
        try:
            val = np.exp(Q[agent_index, state, i] / temperature)
        except OverflowError:
            return i
        
        action_probs_numes.append(val)
        denom += val
    
    action_probs = [x / denom for x in action_probs_numes]
    return np.random.choice(n_actions, 1, p=action_probs)[0]


def run_simulation_for_temperature(scenario, args, temperature):
    """
    Run a single simulation for a given temperature and return the equilibrium order parameter.
    
    Args:
    scenario (Scenario): Loaded scenario object
    args (argparse.Namespace): Simulation parameters
    temperature (float): Temperature for the simulation
    
    Returns:
    float: Equilibrium order parameter
    """
    # Seed for reproducibility
    np.random.seed(13)
    
    # Setup environment
    env, n_agents, n_states, n_actions, dim_Q_state = setup_environment(args, scenario)
    
    # Reward target matrix
    reward_target = np.array([
        [2, -2],
        [1, -1],
        [0, 0],
        [-1, 1],
        [-2, 2]
    ])
    
    # Reset environment
    obs = np.stack(env.reset())
    
    # Initialize Q-values
    Q = np.zeros((n_agents, dim_Q_state, n_actions))
    
    # Track maximum order parameter
    max_order = 0.0
    
    # Fixed simulation parameters
    current_t = temperature
    decay_rate = 0.99
    decay_gap = 2000
    learning_rate = args.learning_rate
    act_rate = args.act_rate
    
    # Main simulation loop
    for t in range(args.time_steps):
        # Decay temperature (optional, can be commented out if not needed)
        if t % decay_gap == 0:
            current_t *= decay_rate
        
        # Select actions for all agents using Boltzmann exploration
        action = np.array([
            boltzmann_exploration(Q, current_t, np.count_nonzero(obs[i] == 1), i, n_actions)
            for i in range(n_agents)
        ])
        
        # Take step in environment
        obs_, reward, done, order_param, ups, downs = env.step(np.expand_dims(action, axis=1))
        obs_ = np.stack(obs_)
        
        # Update Q-values
        act_group = np.random.choice(n_agents, int(act_rate * n_agents), replace=False)
        
        for i in act_group:
            obs_flat = np.count_nonzero(obs[i] == 1)
            Q[i, obs_flat, action[i]] += learning_rate * (
                reward[i] - Q[i, obs_flat, action[i]]
            )
        
        obs = obs_
        
        # Track maximum order parameter
        max_order = max(max_order, order_param)
        
        # Optional early stopping if needed
        if t > 1000 and abs(max_order - order_param) < 0.001:
            break
    
    return max_order


def plot_order_parameter_vs_temperature():
    """
    Plot order parameter as a function of temperature.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load scenario
    scenario = ising_model.load(args.scenario).Scenario()
    
    # Temperature range
    temperatures = np.linspace(0.1, 3.0, 30)
    
    # Collect order parameters
    order_parameters = []
    
    # Run simulations for different temperatures
    for temp in temperatures:
        order_param = run_simulation_for_temperature(scenario, args, temp)
        order_parameters.append(order_param)
        print(f"Temperature: {temp}, Order Parameter: {order_param}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, order_parameters, 'bo-')
    plt.xlabel('Temperature', fontsize=12)
    plt.ylabel('Order Parameter', fontsize=12)
    plt.title('Order Parameter vs Temperature in Mean Field Q-Learning Ising Model', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_folder = "./ising_figs/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, 'order_parameter_vs_temperature.png'))
    plt.close()

    # Save data for potential further analysis
    np.savetxt(os.path.join(output_folder, 'order_parameter_data.csv'), 
               np.column_stack((temperatures, order_parameters)), 
               delimiter=',', 
               header='Temperature,OrderParameter')


def main():
    """
    Main entry point for plotting order parameter vs temperature.
    """
    plot_order_parameter_vs_temperature()


if __name__ == "__main__":
    main()