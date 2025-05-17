# base_ai.py
from abc import ABC, abstractmethod
from typing import Dict, Optional
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
        self.game = game
        self.model_name = model_name
        self.ai_id = ai_id
        self.settings = settings or {}
        self.player_number = 1 if self.ai_id == "ai_1" else 2

        # Load configuration
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        self.config = ai_config["ai_configs"].get(player_key, {}).get("params", DEFAULT_AI_PARAMS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup model paths
        pth_dir = Path(self.config.get("model_dir", "models"))
        pth_dir.mkdir(exist_ok=True)
        self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
        self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"

        # Set ability level
        self.ability = min(max(int(self.config.get("ability_level", 1)), 1), 10)

        # Training variables
        self.memory = deque(maxlen=self.config.get("training_params", {}).get("memory_size", 10000))
        self.gamma = self.config.get("training_params", {}).get("gamma", 0.99)
        self.progress_tracker = None  # Must be initialized in derived class
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.config.get("training_params", {}).get("update_target_every", 1000)
        self.reward_calculator = RewardCalculator(game, ai_id=ai_id)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reward_threshold = self.config.get("training_params", {}).get("reward_threshold", 10.0)

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

    def get_move(self, board):
        """Selects a move for the current board."""
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
                log_to_json(
                    "act() returned None, selecting first valid move",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id}
                )
                move = list(valid_moves.keys())[0]

            (start_row, start_col), (end_row, end_col) = move
            log_to_json(
                f"Selected move: {(start_row, start_col, end_row, end_col)}",
                level="INFO",
                extra_data={"ai_id": self.ai_id}
            )
            return (start_row, start_col, end_row, end_col)
        except Exception as e:
            log_to_json(
                f"Error in get_move: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get move: {str(e)}")

    def get_valid_moves(self, board):
        """
        Retrieves valid moves for the current board.

        Args:
            board (np.ndarray): The game board as an 8x8 numpy array.

        Returns:
            dict: Dictionary mapping move tuples to skipped pieces.
        """
        try:
            valid_moves = {}
            player = 1 if self.player_number == 1 else -1
            temp_board = Board(self.game.settings)
            temp_board.board = board.copy()
            for row in range(self.game.board.board_size):
                for col in range(self.game.board.board_size):
                    piece = board[row, col]
                    if piece != 0 and piece * player > 0:
                        moves = get_piece_moves(temp_board, row, col)
                        for (to_row, to_col), skipped in moves.items():
                            valid_moves[((row, col), (to_row, to_col))] = skipped
            return valid_moves
        except Exception as e:
            log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get valid moves: {str(e)}")

    def get_state(self, board):
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
    def act(self, valid_moves):
        """Selects a move from valid moves."""
        pass

    def update(self, move, reward, board_before, board_after):
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
            action = move  # Use full move tuple
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

    def remember(self, state, action, reward, next_state, done):
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
            batch_size = self.config.get("training_params", {}).get("batch_size", 128)
            if len(self.memory) < batch_size:
                return

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
                ((a[0][0] * board_size + a[0][1]) * board_size * board_size + (a[1][0] * board_size + a[1][1]))
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
                self.config.get("training_params", {}).get("gradient_clip", 1.0)
            )
            self.optimizer.step()

            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.config.get("training_params", {}).get("target_update_alpha", 0.01)) * target_param.data +
                    self.config.get("training_params", {}).get("target_update_alpha", 0.01) * policy_param.data
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
                extra_data={"ai_id": self.ai_id, "batch_size": batch_size}
            )
            raise CheckersError(f"Failed to replay experiences: {str(e)}")

    def save_important_experience(self, state, action, reward, next_state, done):
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

    def load_long_term_memory(self):
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
                            exp["action"],
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

    def update_target_network(self):
        """Updates the target network."""
        try:
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

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """Retrieves default metadata for this AI."""
        return getattr(cls, "AI_METADATA", AI_METADATA)