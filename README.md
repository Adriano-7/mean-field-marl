# Mean Field Multi-Agent Reinforcement Learning

Implementation of **MF-Q** and **MF-AC** as presented in the paper [**Mean Field Multi-Agent Reinforcement Learning**](https://arxiv.org/pdf/1802.05438.pdf).

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

### Compiling the Battle Game Environment

1. **Navigate to the Battle Model Directory:**

    ```bash
    cd examples/battle_model
    ```

2. **Run the Build Script:**

    ```bash
    ./build.sh
    ```

    This script compiles the necessary components for the Battle Game environment. Ensure that you have the required build tools installed on your system.

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
