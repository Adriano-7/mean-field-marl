# Mean Field Multi-Agent Reinforcement Learning (MF-MARL)

This repository implements **Mean Field Q-Learning (MF-Q)** and **Mean Field Actor-Critic (MF-AC)**, based on the approach introduced in the paper [**Mean Field Multi-Agent Reinforcement Learning**](https://arxiv.org/pdf/1802.05438.pdf). These algorithms enable scalable learning in multi-agent environments by approximating interactions through mean-field theory.

## Table of Contents
- [Examples](#examples)
- [Code Structure](#code-structure)
- [Installation](#installation)
  - [Setting Up the Conda Environment](#setting-up-the-conda-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Running the Ising Environment](#running-the-ising-environment)
- [Compiling and Running the MAgent Platform](#compiling-and-running-the-magent-platform)
  - [Compiling the Battle Game Environment](#compiling-the-battle-game-environment)
  - [Training Models for the Battle Game](#training-models-for-the-battle-game)
- [Paper Citation](#paper-citation)

## Examples

### Ising Model Example
A **20x20 Ising model** simulation under low temperature settings.

![Ising Model Simulation](resources/line.gif)

### Battle Game Gridworld Example
A **40x40 Battle Game gridworld** with **128 agents**. In the visualization, the blue agents represent **MFQ**, and the red agents represent **IL**.

<img src="resources/battle.gif" width="300" height="300" alt="Battle Game Simulation"/>

## Code Structure

- **`main_MFQ_Ising.py`**: Runs the tabular-based MFQ algorithm for the Ising model.
  
- **`./examples/`**: Contains scenarios for the Ising Model and Battle Game, including the necessary models.
  
- **`battle.py`**: Executes the Battle Game using a trained model.
  
- **`train_battle.py`**: Scripts for training Battle Game models.

## Installation

### Setting Up the Conda Environment

Before compiling and running the project, it's recommended to create a dedicated Conda environment to manage dependencies.

1. **Create a Conda Environment:**

    ```bash
    conda create --name tsi_mfrl python=3.6.1 -c conda-forge
    ```

2. **Activate the Conda Environment:**

    ```bash
    conda activate tsi_mfrl
    ```

### Installing Dependencies

With the Conda environment activated, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

## Running the Ising Environment

### Steps to Run

1. **Navigate to the Project Directory:**

    ```bash
    cd path_to_your_project_directory
    ```

2. **Run the Ising Model Simulation:**

    ```bash
    python main_MFQ_Ising.py
    ```

    This will execute the MFQ algorithm on the Ising model and generate the corresponding figures if `matplotlib` is installed.

## Compiling and Running the MAgent Platform

Before running the Battle Game environment, you need to compile the MAgent platform. Follow the steps below for compilation and execution.

## Compiling the MAgent Platform

Before running the Battle Game environment, you need to compile the MAgent platform, which is already included in this repository.

### Compiling MAgent on Linux

1. **Install Dependencies:**

    ```bash
    sudo apt-get update
    sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
    ```

2. **Build MAgent:**

    ```bash
    cd examples/battle_model
    ./build.sh
    ```

### Compiling MAgent on OSX

**Note:** There is an issue with Homebrew for installing `websocketpp`. Please refer to [Issue #17](https://github.com/geek-ai/MAgent/issues/17) for more details.

1. **Install Dependencies Using Homebrew:**

    ```bash
    brew install cmake llvm boost@1.55
    brew install jsoncpp argp-standalone
    brew tap david-icracked/homebrew-websocketpp
    brew install --HEAD david-icracked/websocketpp/websocketpp
    brew link --force boost@1.55
    ```

2. **Build MAgent:**

    ```bash
    cd examples/battle_model
    ./build.sh
    ```

### Training Models for the Battle Game

#### 1. Add Python Path

To ensure that Python can locate the necessary modules, add the Battle Model's Python directory to your `PYTHONPATH`. You can do this by adding the following lines to your shell configuration file (`~/.bashrc` or `~/.zshrc`):

```bash
export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
source ~/.bashrc  # or source ~/.zshrc
```

#### 2. Run the Training Script

To train models using a specific algorithm (e.g., **MFAC**), execute the training script with the appropriate arguments:

```bash
python3 train_battle.py --algo mfac
```

**Additional Options:**

To view all available command-line options and arguments, use the `--help` flag:

```bash
python3 train_battle.py --help
```

This will display detailed information about the various parameters you can set during training, such as `--save_every`, `--update_every`, `--n_round`, `--render`, `--map_size`, and `--max_steps`.

Here’s the updated section for your `battle.py` script in the README:

---

### Running the Battle Game Using Trained Models

Once you have trained models for the Battle Game, you can use the `battle.py` script to simulate a match between two agents, with one agent using the **main model** and the other using the **opponent model**.

#### Running the Battle Script

To run the script and simulate battles between the agents:

```bash
python3 battle.py --algo ac --oppo mfq --n_round 50 --map_size 40 --max_steps 400 --idx 100 200
```

- `--algo`: Specifies the algorithm used by the **main agent** (e.g., `ac`, `mfac`, `mfq`, or `il`).
- `--oppo`: Specifies the algorithm used by the **opponent agent** (e.g., `ac`, `mfac`, `mfq`, or `il`).
- `--n_round`: The number of rounds (games) to simulate.
- `--map_size`: Size of the gridworld map (default is 40x40).
- `--max_steps`: Maximum number of steps per round.
- `--idx`: Model checkpoints to load for both agents (default is `[100, 200]`).

## Paper Citation

If you find this project helpful in your research or work, please consider citing the following paper:

```bibtex
@InProceedings{pmlr-v80-yang18d,
  title = 	 {Mean Field Multi-Agent Reinforcement Learning},
  author = 	 {Yang, Yaodong and Luo, Rui and Li, Minne and Zhou, Ming and Zhang, Weinan and Wang, Jun},
  booktitle = 	 {Proceedings of the 35th International Conference on Machine Learning},
  pages = 	 {5567--5576},
  year = 	 {2018},
  editor = 	 {Dy, Jennifer and Krause, Andreas},
  volume = 	 {80},
  series = 	 {Proceedings of Machine Learning Research},
  address = 	 {Stockholmsmässan, Stockholm Sweden},
  month = 	 {10--15 Jul},
  publisher = 	 {PMLR}
}
```

---

