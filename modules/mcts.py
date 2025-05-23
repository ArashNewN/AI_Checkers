import os
import numpy as np
import torch
import torch.multiprocessing as mp
from collections import defaultdict
import logging
import hashlib
import json
import gzip
from contextlib import closing
from time import time

from pyfile.config import load_ai_config

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

log_dir = os.path.dirname(os.path.dirname(__file__))
log_file = os.path.join(log_dir, 'mcts.log')
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)

class MCTS:
    def __init__(self, ai, game, neural_net):
        self.config = self._load_config()
        mcts_params = self.config["mcts_params"]
        self.ai = ai
        self.game = game
        self.neural_net = neural_net
        self.c_puct = mcts_params["c_puct"]
        self.num_simulations = mcts_params["num_simulations"]
        self.max_cache_size = mcts_params["max_cache_size"]
        self.num_processes = mcts_params["num_processes"]
        self.cache_save_interval = mcts_params["cache_save_interval"]
        self.search_count = 0
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.P = defaultdict(dict)
        self.children = defaultdict(list)
        self.device = next(neural_net.parameters()).device if neural_net else torch.device('cpu')
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, mcts_params["cache_file"])
        self.state_cache = self._load_cache()
        if torch.cuda.is_available():
            mp.set_sharing_strategy('file_system')
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")

    def _load_config(self):
        """بارگذاری و اعتبارسنجی تنظیمات MCTS از ai_config.json"""
        config = load_ai_config()
        if "mcts_params" not in config:
            raise KeyError("Missing mcts_params in ai_config.json")

        mcts_params = config["mcts_params"]
        required_params = [
            "c_puct", "num_simulations", "max_cache_size",
            "num_processes", "cache_save_interval", "cache_file"
        ]
        for param in required_params:
            if param not in mcts_params:
                raise KeyError(f"Missing {param} in mcts_params in ai_config.json")

        return {"mcts_params": mcts_params}

    def _load_cache(self):
        logger.info(f"Attempting to load cache from {self.cache_file}")
        try:
            with gzip.open(self.cache_file, "rt", encoding="utf-8") as f:
                cache = json.load(f)
                logger.info(f"Cache loaded with {len(cache)} entries")
                return {k: np.array(v) for k, v in cache.items()}
        except (FileNotFoundError, json.JSONDecodeError, EOFError):
            logger.info(f"No existing cache found or error loading cache")
            return {}

    def _save_cache(self):
        logger.info(f"Saving cache to {self.cache_file}")
        try:
            cache_to_save = {k: v.tolist() for k, v in self.state_cache.items()}
            with gzip.open(self.cache_file, "wt", encoding="utf-8") as f:
                json.dump(cache_to_save, f)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _simulate_wrapper(self, args):
        try:
            state_array, state_key = args
            state = self.game.copy()
            state.set_state(state_array)
            return self._simulate(state, state_key)
        except Exception as e:
            logger.error(f"Error in simulate_wrapper: {e}")
            return None

    def search(self, state):
        start_time = time()
        logger.debug("Entering search method")
        self.search_count += 1
        state_bytes = state.get_state().tobytes()
        root_key = hashlib.md5(state_bytes).hexdigest()
        logger.debug(f"State key: {root_key}")
        if root_key in self.state_cache:
            logger.info("Cache hit for state")
            logger.debug(f"Search completed in {time() - start_time:.2f} seconds")
            return self.state_cache[root_key]
        legal_moves = state.get_legal_moves()
        logger.debug(f"Legal moves: {legal_moves}")
        if not legal_moves:
            logger.warning("No legal moves available")
            logger.debug(f"Search completed in {time() - start_time:.2f} seconds")
            return np.zeros(self.game.get_action_size())
        logger.info(f"Searching for state with {len(legal_moves)} legal moves")
        if torch.cuda.is_available() and self.num_processes > 1 and self.neural_net:
            logger.info(f"Parallel simulation with {self.num_processes} processes on GPU")
            try:
                state_array = state.get_state()
                tasks = [(state_array, root_key) for _ in range(self.num_simulations)]
                simulations_per_process = self.num_simulations // self.num_processes
                with closing(mp.Pool(processes=self.num_processes)) as pool:
                    results = pool.map(self._simulate_wrapper, tasks[:self.num_processes * simulations_per_process])
                for _ in range(self.num_simulations % self.num_processes):
                    self._simulate(state.copy(), root_key)
            except Exception as e:
                logger.error(f"Parallel simulation failed: {e}, falling back to sequential")
                for _ in range(self.num_simulations):
                    self._simulate(state.copy(), root_key)
        else:
            logger.info("Running sequential simulation")
            for _ in range(self.num_simulations):
                self._simulate(state.copy(), root_key)

        move_probs = np.zeros(self.game.get_action_size(), dtype=np.float32)
        total_visits = sum(self.N[child_key] for _, child_key in self.children[root_key])
        for move, child_key in self.children[root_key]:
            if move in legal_moves:
                from_idx, to_idx = move
                if 0 <= from_idx < 32 and 0 <= to_idx < 32 and from_idx != to_idx:
                    idx = from_idx * 32 + to_idx
                    move_probs[idx] = self.N[child_key] / total_visits if total_visits > 0 else 0.0
        if np.sum(move_probs) == 0:
            logger.warning("No valid move probabilities, using uniform distribution")
            for move in legal_moves:
                from_idx, to_idx = move
                if 0 <= from_idx < 32 and 0 <= to_idx < 32 and from_idx != to_idx:
                    idx = from_idx * 32 + to_idx
                    move_probs[idx] = 1.0 / len(legal_moves)
        else:
            move_probs /= np.sum(move_probs)

        self.state_cache[root_key] = move_probs
        if len(self.state_cache) > self.max_cache_size:
            oldest_key = next(iter(self.state_cache))
            del self.state_cache[oldest_key]
        if self.search_count % self.cache_save_interval == 0:
            self._save_cache()
        logger.info(f"Search completed in {time() - start_time:.2f} seconds")
        logger.info(f"Returning move_probs: shape={move_probs.shape}, sum={np.sum(move_probs)}")
        return move_probs

    def _simulate(self, state, state_key):
        logger.debug(f"Simulating state: {state_key}")
        if state_key not in self.children:
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                value = state.get_outcome()
                logger.warning(f"No legal moves in simulate, outcome: {value}")
                return -value
            if not self.neural_net:
                logger.error("Neural network not provided")
                raise ValueError("Neural network required")
            converted_state = self.ai.convert_state_to_network_format(state.get_state())
            state_tensor = torch.FloatTensor(converted_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy, value = self.neural_net(state_tensor, legal_moves=legal_moves)
            policy = policy[0].detach().cpu().numpy()
            value = value[0].item()
            logger.debug(f"Policy output: {policy[:10]}...")
            valid_policy = {}
            valid_children = []
            for move in legal_moves:
                from_idx, to_idx = move
                if not (0 <= from_idx < 32 and 0 <= to_idx < 32 and from_idx != to_idx):
                    logger.warning(f"Skipping invalid move indices: {move}")
                    continue
                idx = from_idx * 32 + to_idx
                valid_policy[move] = policy[idx] if idx < len(policy) else 0.0
                try:
                    child_state = state.copy().make_move(move)
                    child_key = hashlib.md5(child_state.get_state().tobytes()).hexdigest()
                    valid_children.append((move, child_key))
                except ValueError as e:
                    logger.warning(f"Invalid move {move}: {e}")
                    continue
            if not valid_children:
                logger.warning("No valid children for state")
                return -state.get_outcome()
            self.P[state_key] = valid_policy
            self.children[state_key] = valid_children
            return -value

        valid_children = [(move, child_key) for move, child_key in self.children[state_key]
                         if move in self.P[state_key]]
        if not valid_children:
            logger.warning("No valid children for state")
            return -state.get_outcome()

        best_move, best_child = max(
            valid_children,
            key=lambda x: self.Q[x[1]] + self.c_puct * self.P[state_key][x[0]] * np.sqrt(self.N[state_key]) / (1 + self.N[x[1]])
        )
        logger.debug(f"Selected move: {best_move}")
        next_state = state.copy().make_move(best_move)
        value = -self._simulate(next_state, best_child)
        self.Q[best_child] = (self.N[best_child] * self.Q[best_child] + value) / (self.N[best_child] + 1)
        self.N[best_child] += 1
        self.N[state_key] += 1
        return value