import json
import os

DEFAULT_LEARN_CONFIG = {
    "num_episodes_az1": 1,
    "num_games_az1": 1,
    "mcts_simulations_az1": 100,
    "batch_size_az1": 32,
    "learning_rate_az1": 0.001,
    "c_puct_az1": 1.5,
    "memory_size_az1": 100000,
    "max_moves_az1": 100,
    "num_episodes_az2": 1,
    "num_games_az2": 1,
    "mcts_simulations_az2": 100,
    "batch_size_az2": 32,
    "learning_rate_az2": 0.001,
    "c_puct_az2": 1.5,
    "memory_size_az2": 100000,
    "max_moves_az2": 100
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "learn_method.json")

def load_learn_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)
                return {**DEFAULT_LEARN_CONFIG, **config}
        except (json.JSONDecodeError, IOError):
            pass
    return DEFAULT_LEARN_CONFIG

def save_learn_config(config):
    try:
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file, indent=4)
    except IOError as e:
        print(f"Error saving learn config: {e}")