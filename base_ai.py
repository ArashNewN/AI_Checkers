#base_ai.py
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from .rewards import RewardCalculator
from .checkers_core import log_to_json
from .config import load_ai_config, DEFAULT_AI_PARAMS
from .utils import CheckersError

# Default metadata for BaseAI
AI_METADATA = {
    "default_type": "base_ai",
    "default_description": "Base AI with fundamental capabilities for the checkers game."
}


class BaseAI(ABC):
    """Abstract base class for AI implementations in the checkers game."""

    def __init__(self, game, model_name: str, ai_id: str, settings: Optional[Dict] = None):
        """Initializes the base AI."""
        if not hasattr(game, 'copy') or not hasattr(game, 'get_legal_moves'):
            raise ValueError(f"Provided game object does not support required methods (copy, get_legal_moves)")
        self.game = game
        self.model_name = model_name
        self.ai_id = ai_id
        self.settings = settings or {}
        self.player_number = 1 if self.ai_id == "ai_1" else 2
        self.player = 1 if self.player_number == 1 else -1

        # Load configuration
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        self.config = ai_config["ai_configs"].get(player_key, {}).get("params", DEFAULT_AI_PARAMS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model paths
        pth_dir = Path(self.config["model_dir"])
        pth_dir.mkdir(exist_ok=True)
        self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
        self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"

        # Set ability level
        self.ability = min(max(int(self.config["ability_level"]), 1), 10)

        # Training variables
        self.memory = deque(maxlen=self.config["training_params"]["memory_size"])
        self.gamma = self.config["training_params"]["gamma"]
        self.progress_tracker = None  # Must be initialized in derived class
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.config["training_params"]["update_target_every"]
        self.reward_calculator = RewardCalculator(game, ai_id=ai_id)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reward_threshold = self.config["training_params"]["reward_threshold"]

        # Neural network and optimizer (to be initialized in derived classes)
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        log_to_json(
            f"Initialized BaseAI for {ai_id} with model {model_name}",
            level="INFO",
            extra_data={"device": str(self.device), "ability": self.ability}
        )

    def _default_config(self):
        """Provides default configuration for AI."""
        return {
            "model_dir": "models",
            "ability_level": 1,
            "training_params": {
                "memory_size": 10000,
                "gamma": 0.99,
                "batch_size": 128,
                "update_target_every": 1000,
                "reward_threshold": 10.0,
                "gradient_clip": 1.0,
                "target_update_alpha": 0.01
            },
            "network_params": {
                "board_size": 8
            },
            "advanced_nn_params": {
                "input_channels": 3
            }
        }

    def set_game(self, game):
        """Updates the game object and reward calculator."""
        self.game = game
        if self.reward_calculator:
            self.reward_calculator.game = game
        log_to_json(f"Updated game object for {self.ai_id}", level="INFO")

    def get_move(self, board: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Selects a move for the current board in 4-tuple format."""
        try:
            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                log_to_json(
                    "No valid moves available",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id, "board": board.tolist()}
                )
                return None

            move = self.act(valid_moves)
            if not move:
                move = valid_moves[0]
                log_to_json(
                    "act() returned None, selecting first valid move",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id}
                )

            log_to_json(
                f"Selected move: {move}",
                level="INFO",
                extra_data={"ai_id": self.ai_id}
            )
            return move
        except Exception as e:
            log_to_json(
                f"Error in get_move: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get move: {str(e)}")

    def get_valid_moves(self, board: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Retrieves valid moves from current board using game API (4-tuple format)."""
        try:
            temp_game = self.game.copy()
            temp_game.set_state(board)
            temp_game.current_player = self.player
            return temp_game.get_legal_moves()
        except Exception as e:
            log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get valid moves: {str(e)}")

    def get_state(self, board: np.ndarray) -> torch.Tensor:
        """Converts the game board to a state input for the neural network."""
        try:
            board_size = self.game.board.board_size
            input_channels = self.config.get("advanced_nn_params", {}).get("input_channels", 3)
            state = np.zeros((input_channels, board_size, board_size), dtype=np.float32)
            for row in range(board_size):
                for col in range(board_size):
                    piece = board[row, col]
                    if piece != 0:
                        if piece > 0:
                            state[0, row, col] = 1  # Player 1 pieces
                            if abs(piece) == 2:
                                state[2, row, col] = 1  # Kings
                        else:
                            state[1, row, col] = 1  # Player 2 pieces
                            if abs(piece) == 2:
                                state[2, row, col] = 1  # Kings
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
        except Exception as e:
            log_to_json(
                f"Error in get_state: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get state: {str(e)}")

    @abstractmethod
    def act(self, valid_moves: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Selects a move from valid moves in 4-tuple format."""
        pass

    def update(self, move: Tuple[int, int, int, int], reward: float, board_before: np.ndarray, board_after: np.ndarray):
        """Updates the AI model based on the move and received reward."""
        try:
            if move is None:
                log_to_json(
                    "Move is None, skipping update",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id}
                )
                return

            state = self.get_state(board_before)
            next_state = self.get_state(board_after)
            action = move
            done = self.game.game_over

            self.remember(state, action, reward, next_state, done)
            self.replay()
            self.update_target_network()
            self.save_model()
        except Exception as e:
            log_to_json(
                f"Error in update: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "move": move}
            )
            raise CheckersError(f"Failed to update AI: {str(e)}")

    def remember(self, state: torch.Tensor, action: Tuple[int, int, int, int], reward: float, next_state: torch.Tensor,
                 done: bool):
        """Stores experience in memory."""
        try:
            self.current_episode_reward += reward
            self.memory.append((state, action, reward, next_state, done))
            self.save_important_experience(state, action, reward, next_state, done)
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                    json.dump(self.episode_rewards, f)
        except Exception as e:
            log_to_json(
                f"Error in remember: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "action": action}
            )
            raise CheckersError(f"Failed to store experience: {str(e)}")

    def replay(self):
        """Trains the model using stored experiences."""
        try:
            if "training_params" not in self.config or "batch_size" not in self.config["training_params"]:
                raise CheckersError("Missing 'batch_size' in training_params configuration")
            batch_size = self.config["training_params"]["batch_size"]
            if len(self.memory) < batch_size:
                return

            if self.policy_net is None or self.target_net is None or self.optimizer is None:
                raise CheckersError("Neural networks or optimizer not initialized")

            batch = list(self.memory)[-batch_size:]
            long_term_memory = self.load_long_term_memory()
            if long_term_memory:
                long_term_batch = sorted(long_term_memory, key=lambda x: x[2], reverse=True)[:batch_size // 2]
                batch = batch[:(batch_size // 2)] + long_term_batch

            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states)
            next_states = torch.cat(next_states)
            board_size = self.game.board.board_size
            action_indices = torch.tensor([
                (a[0] * board_size + a[1]) * (board_size * board_size) + (a[2] * board_size + a[3])
                for a in actions if a is not None
            ], device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            current_q_values = self.policy_net(states).gather(1, action_indices.unsqueeze(1))
            with torch.no_grad():
                max_next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config["training_params"]["gradient_clip"]
            )
            self.optimizer.step()

            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.config["training_params"]["target_update_alpha"]) * target_param.data +
                    self.config["training_params"]["target_update_alpha"] * policy_param.data
                )

            self.epoch += 1
            policy_loss = loss.item()
            player1_wins = sum(1 for r in self.episode_rewards if r > 0)
            player2_wins = sum(1 for r in self.episode_rewards if r < 0)
            self.progress_tracker.log_epoch(self.epoch, policy_loss, policy_loss, player1_wins, player2_wins)
        except Exception as e:
            log_to_json(
                f"Error in replay: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "batch_size": locals().get("batch_size", "undefined")}
            )
            raise CheckersError(f"Failed to replay experiences: {str(e)}")

    def update_target_network(self):
        """Updates the target network."""
        try:
            if self.policy_net is None or self.target_net is None:
                raise CheckersError("Neural networks not initialized")

            self.episode_count += 1
            if self.episode_count % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                log_to_json(
                    f"Updated target network for {self.ai_id}",
                    level="INFO",
                    extra_data={"episode_count": self.episode_count}
                )
        except Exception as e:
            log_to_json(
                f"Error updating target network: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id}
            )
            raise CheckersError(f"Failed to update target network: {str(e)}")

    def save_model(self):
        """Saves the model and episode rewards."""
        try:
            if self.policy_net is None:
                raise CheckersError("Policy network not initialized")

            torch.save(self.policy_net.state_dict(), self.model_path)
            torch.save(self.policy_net.state_dict(), self.backup_path)
            with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                json.dump(self.episode_rewards, f)
            log_to_json(
                f"Saved model for {self.ai_id}",
                level="INFO",
                extra_data={"model_path": str(self.model_path)}
            )
        except Exception as e:
            log_to_json(
                f"Error saving model: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "model_path": str(self.model_path)}
            )
            raise CheckersError(f"Failed to save model: {str(e)}")

    def save_important_experience(self, state: torch.Tensor, action: Tuple[int, int, int, int], reward: float,
                                  next_state: torch.Tensor, done: bool):
        """Stores important experiences in long-term memory."""
        try:
            if reward > self.reward_threshold:
                with open(self.long_term_memory_path, "a") as f:
                    experience = {
                        "state": state.cpu().numpy().tolist(),
                        "action": action,
                        "reward": float(reward),
                        "next_state": next_state.cpu().numpy().tolist(),
                        "done": done,
                    }
                    json.dump(experience, f)
                    f.write("\n")
        except Exception as e:
            log_to_json(
                f"Error saving important experience: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "reward": reward}
            )
            raise CheckersError(f"Failed to save experience: {str(e)}")

    def load_long_term_memory(self) -> List[Tuple[torch.Tensor, Tuple[int, int, int, int], float, torch.Tensor, bool]]:
        """Loads experiences from long-term memory."""
        try:
            if os.path.exists(self.long_term_memory_path):
                with open(self.long_term_memory_path, "r") as f:
                    experiences = []
                    for line in f.readlines():
                        exp = json.loads(line)
                        exp["state"] = torch.FloatTensor(exp["state"]).to(self.device)
                        exp["next_state"] = torch.FloatTensor(exp["next_state"]).to(self.device)
                        experiences.append((
                            exp["state"],
                            tuple(exp["action"]),  # Ensure action is a tuple
                            exp["reward"],
                            exp["next_state"],
                            exp["done"]
                        ))
                    return experiences
            return []
        except Exception as e:
            log_to_json(
                f"Error loading long-term memory: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "path": str(self.long_term_memory_path)}
            )
            raise CheckersError(f"Failed to load long-term memory: {str(e)}")

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """Retrieves default metadata for this AI."""
        return getattr(cls, "AI_METADATA", AI_METADATA)
