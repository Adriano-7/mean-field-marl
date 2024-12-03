import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import examples.ising_model as ising_model
from examples.ising_model.multiagent.environment import IsingMultiAgentEnv

np.random.seed(13)

def parse_arguments():
    """
    Parse command-line arguments for the Ising Model simulation.
    """
    parser = argparse.ArgumentParser(description="Ising Model Multi-Agent Reinforcement Learning Simulation")
    parser.add_argument('-n', '--num_agents', default=100, type=int)
    parser.add_argument('-epi', '--episode', default=1, type=int)
    parser.add_argument('-t', '--temperature', default=1, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float)
    parser.add_argument('-dr', '--decay_rate', default=0.99, type=float)
    parser.add_argument('-dg', '--decay_gap', default=2000, type=int)
    parser.add_argument('-ac', '--act_rate', default=1.0, type=float)
    parser.add_argument('-ns', '--neighbor_size', default=4, type=int)
    parser.add_argument('-s', '--scenario', default='Ising.py', help='Path of the scenario Python script.')
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


def MFQ_Simulation(scenario, args, temperature, time_steps=10000, save_plot=False, folder=None):
    """
    Run a single simulation for a given temperature and return the equilibrium order parameter.
    
    Args:
    scenario (Scenario): Loaded scenario object
    args (argparse.Namespace): Simulation parameters
    temperature (float): Temperature for the simulation
    
    Returns:
    float: Equilibrium order parameter
    """

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
    mse = 0

    if save_plot:
        plt.figure(2)
        plt.ion()
        ising_plot = np.zeros((int(np.sqrt(n_agents)), int(np.sqrt(n_agents))), dtype=np.int32)
        im = plt.imshow(ising_plot, cmap='gray', vmin=0, vmax=1, interpolation='none')
        im.set_data(ising_plot)


    
    # Fixed simulation parameters
    current_t = temperature
    
    # Main simulation loop
    for t in range(time_steps):
        # Decay temperature (optional, can be commented out if not needed)
        if t % args.decay_gap == 0:
            current_t *= args.decay_rate
        
        # Select actions for all agents using Boltzmann exploration
        action = np.array([
            boltzmann_exploration(Q, current_t, np.count_nonzero(obs[i] == 1), i, n_actions)
            for i in range(n_agents)
        ])
        
        display = action.reshape((int(np.sqrt(n_agents)), -1))

        # Take step in environment
        obs_, reward, done, order_param, ups, downs = env.step(np.expand_dims(action, axis=1))
        obs_ = np.stack(obs_)
        
        # Update Q-values
        mse=0
        act_group = np.random.choice(n_agents, int(args.act_rate * n_agents), replace=False)
        
        for i in act_group:
            obs_flat = np.count_nonzero(obs[i] == 1)
            Q[i, obs_flat, action[i]] += args.learning_rate * (
                reward[i] - Q[i, obs_flat, action[i]]
            )
            mse += np.power((Q[i, obs_flat, action[i]] - reward_target[obs_flat, action[i]]), 2)
        
        mse /= n_agents
        obs = obs_

        # Track maximum order parameter
        if save_plot and order_param > max_order:
            plt.figure(2)
            im.set_data(display)
            plt.savefig(folder + f'{t}.png')

        max_order = max(max_order, order_param)
            
        # Optional early stopping if needed
        if t > 1000 and abs(max_order - order_param) < 0.001:
            break
    
    return max_order, mse


def plot_order_parameter_vs_temperature():
    """
    Plot order parameter as a function of temperature.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load scenario
    scenario = ising_model.load(args.scenario).Scenario()
    
    # Temperature range
    temperatures = np.linspace(0.1, 3.0, 20)
    
    # Collect order parameters
    order_parameters = []
    
    # Run simulations for different temperatures
    for temp in temperatures:
        order_param = MFQ_Simulation(scenario, args, temp)[0]
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

def plot_op_and_mse_vs_timestep():
    """
    Plot order parameter and mse as a function of temperature.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load scenario
    scenario = ising_model.load(args.scenario).Scenario()
    
    # Temperature range
    timesteps = np.linspace(1, 1500, 100)
    
    # Collect order parameters
    order_parameters = []
    mses = []
    
    # Run simulations for different temperatures
    for timestep in timesteps:
        timestep = int(timestep)
        order_param, mse = MFQ_Simulation(scenario, args, args.temperature, timestep)
        order_parameters.append(order_param)
        mses.append(mse)
        print(f"Timestep: {timestep}, Order Parameter: {order_param}, MSE: {mse}")
    
    # Plot the results (order param vs timesteps) with the green line being the mse and blue line being the order parameter
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, order_parameters, 'bo-')
    plt.plot(timesteps, mses, 'go-')
    plt.xlabel('Timesteps', fontsize=12)
    plt.ylabel('Order Parameter/MSE', fontsize=12)
    plt.title('Order Parameter and MSE vs Timesteps in Mean Field Q-Learning Ising Model', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_folder = "./ising_figs/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, f'order_parameter_and_mse_vs_timesteps_T{temp}.png'))
    plt.close()

def plot_simulation_states():
    """
    Plot the states of the simulation.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Load scenario
    scenario = ising_model.load(args.scenario).Scenario()
    temperature = args.temperature
    
    folder = f"./ising_figs/gifs/Temp_{temperature}_Time_{time.strftime('%Y%m%d-%H%M%S')}/"
    os.makedirs(folder)
    
    MFQ_Simulation(scenario, args, temperature, time_steps=1200, save_plot=True, folder=folder)

    import imageio
    import glob

    images = []
    for filename in sorted(glob.glob(f"{folder}/*.png")):
        images.append(imageio.imread(filename))
    imageio.mimsave(f"{folder}/animation.gif", images, duration=0.1)
    

if __name__ == "__main__":
    """
    Main entry point for plotting order parameter vs temperature.
    """
    #plot_order_parameter_vs_temperature()
    #plot_op_and_mse_vs_timestep()
    plot_simulation_states()
