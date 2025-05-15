from abc import ABC, abstractmethod
from typing import Dict, Optional
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from pathlib import Path

# متادیتای پیش‌فرض برای BaseAI
AI_METADATA = {
    "default_type": "base_ai",
    "default_description": "هوش مصنوعی پایه با قابلیت‌های اولیه برای بازی شطرنج."
}

class BaseAI(ABC):
    """کلاس پایه انتزاعی برای هوش‌های مصنوعی در بازی شطرنج."""

    def __init__(self, game, model_name: str, ai_id: str, settings: Optional[Dict] = None):
        """
        مقداردهی اولیه هوش مصنوعی پایه.

        Args:
            game: شیء بازی (CheckersGame)
            model_name: نام مدل برای ذخیره/بارگذاری
            ai_id: شناسه منحصربه‌فرد برای AI (مثل 'ai_1' یا 'ai_2')
            settings: تنظیمات اختیاری از config.json
        """
        self.game = game
        self.model_name = model_name
        self.ai_id = ai_id
        self.settings = settings or {}
        self.player_number = 1 if self.ai_id == "ai_1" else 2

        # تنظیمات و مسیرها
        self.config = self._load_config(settings, ai_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pth_dir = Path(self.config["model_dir"])
        pth_dir.mkdir(exist_ok=True)
        self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
        self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"

        # تنظیم سطح توانایی
        self.ability = min(max(int(self.config["ability_level"]), 1), 10)

        # متغیرهای آموزش
        self.memory = deque(maxlen=self.config["training_params"]["memory_size"])
        self.gamma = self.config["training_params"]["gamma"]
        self.progress_tracker = None  # باید در کلاس مشتق‌شده مقداردهی شود
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.config["training_params"]["update_target_every"]
        self.reward_calculator = None  # باید در کلاس مشتق‌شده مقداردهی شود
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reward_threshold = self.config["training_params"]["reward_threshold"]

    def _load_config(self, config, ai_id):
        """بارگذاری تنظیمات پیش‌فرض. کلاس‌های مشتق‌شده می‌توانند این متد را بازنویسی کنند."""
        return config or {}

    def set_game(self, game):
        self.game = game
        if self.reward_calculator:
            self.reward_calculator.game = game

    def get_move(self, board):
        """
        انتخاب حرکت برای تخته فعلی.

        Args:
            board: وضعیت فعلی تخته بازی

        Returns:
            حرکت انتخاب‌شده در فرمت (start_row, start_col, end_row, end_col)
        """
        print(f"BaseAI.get_move called for {self.ai_id} (player_number: {self.player_number})")
        valid_moves = self.get_valid_moves(board)
        print(f"Valid moves: {valid_moves}")
        if not valid_moves:
            print("Warning: No valid moves available in BaseAI.get_move")
            return None

        move = self.act(valid_moves)
        print(f"Selected move by act: {move}")
        if not move:
            print("Warning: act() returned None, selecting first valid move")
            move = list(valid_moves.keys())[0]

        # تبدیل کلید حرکت به فرمت (start_row, start_col, end_row, end_col)
        (start_row, start_col), (end_row, end_col) = move
        return (start_row, start_col, end_row, end_col)

    def get_valid_moves(self, board):
        """
        دریافت حرکت‌های معتبر برای تخته فعلی.

        Args:
            board: وضعیت فعلی تخته بازی

        Returns:
            دیکشنری حرکت‌های معتبر در فرمت {((start_row, start_col), (end_row, end_col)): skipped}
        """
        valid_moves = {}
        board_size = self.config["network_params"].get("board_size", 8)
        print(f"Getting valid moves for {self.ai_id} (player_number: {self.player_number}) with board shape: {board.shape}")

        for row in range(board_size):
            for col in range(board_size):
                piece = board[row, col]
                if piece != 0 and (
                        (piece > 0 and self.player_number == 1) or (piece < 0 and self.player_number == 2)
                ):
                    print(f"Checking piece at ({row}, {col}): {piece}")
                    moves = self.game.get_valid_moves(row, col)
                    print(f"Moves for ({row}, {col}): {moves}")
                    if moves:
                        for move, skipped in moves.items():
                            if move[:2] == (row, col):  # فقط حرکات شروع‌شده از (row, col)
                                valid_moves[((row, col), (move[2], move[3]))] = skipped
                        print(f"Added moves for ({row}, {col}): {valid_moves}")

        print(f"Final valid moves: {valid_moves}")
        return valid_moves

    def get_state(self, board):
        """
        تبدیل تخته بازی به حالت ورودی برای شبکه عصبی.

        Args:
            board: وضعیت فعلی تخته بازی

        Returns:
            تنسور حالت بازی
        """
        board_size = self.config["network_params"].get("board_size", 8)
        input_channels = self.config["advanced_nn_params"].get("input_channels", 3)
        state = np.zeros((input_channels, board_size, board_size))
        for row in range(board_size):
            for col in range(board_size):
                piece = board[row, col]
                if piece != 0:
                    if piece > 0:
                        state[0, row, col] = 1
                        if abs(piece) == 2:
                            state[2, row, col] = 1
                    else:
                        state[1, row, col] = 1
                        if abs(piece) == 2:
                            state[2, row, col] = 1
        print(f"State created with shape: {state.shape}")
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    @abstractmethod
    def act(self, valid_moves):
        """
        انتخاب حرکت از بین حرکت‌های معتبر.

        Args:
            valid_moves: دیکشنری حرکت‌های معتبر

        Returns:
            حرکت انتخاب‌شده
        """
        pass

    def update(self, move, reward, board_before, board_after):
        """
        به‌روزرسانی مدل AI بر اساس حرکت و پاداش دریافتی.

        Args:
            move: حرکت انجام‌شده
            reward: پاداش دریافتی
            board_before: تخته قبل از حرکت
            board_after: تخته بعد از حرکت
        """
        print(f"BaseAI.update called for {self.ai_id} with move: {move}, reward: {reward}")
        if move is None:
            print("Warning: Move is None, skipping update")
            return

        state = self.get_state(board_before)
        next_state = self.get_state(board_after)
        action = move[1] if isinstance(move, tuple) and len(move) == 2 else None
        done = self.game.game_over

        if action is None:
            print("Warning: Invalid action, skipping update")
            return

        self.remember(state, action, reward, next_state, done)
        self.replay()
        self.update_target_network()
        self.save_model()

    def remember(self, state, action, reward, next_state, done):
        """
        ذخیره تجربه در حافظه.

        Args:
            state: حالت فعلی
            action: حرکت انجام‌شده
            reward: پاداش دریافتی
            next_state: حالت بعدی
            done: آیا بازی تمام شده است
        """
        print(f"Remembering experience: reward={reward}, done={done}")
        self.current_episode_reward += reward
        self.memory.append((state, action, reward, next_state, done))
        self.save_important_experience(state, action, reward, next_state, done)
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                json.dump(self.episode_rewards, f)

    def replay(self):
        """
        آموزش مدل با استفاده از تجربه‌های ذخیره‌شده.
        """
        batch_size = self.config["training_params"].get("batch_size", 128)
        if len(self.memory) < batch_size:
            print(f"Memory size {len(self.memory)} is less than batch size {batch_size}, skipping replay")
            return

        batch = list(self.memory)[-batch_size:]
        long_term_memory = self.load_long_term_memory()
        if long_term_memory:
            long_term_batch = sorted(long_term_memory, key=lambda x: x[2], reverse=True)[:batch_size // 2]
            batch = batch[:(batch_size // 2)] + long_term_batch

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        board_size = self.config["network_params"].get("board_size", 8)
        actions = torch.tensor([a[0] * board_size + a[1] for a in actions if a is not None], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config["training_params"].get("gradient_clip", 1.0))
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                (1 - self.config["training_params"].get("target_update_alpha", 0.01)) * target_param.data +
                self.config["training_params"].get("target_update_alpha", 0.01) * policy_param.data
            )

        self.epoch += 1
        policy_loss = loss.item()
        player1_wins = sum(1 for r in self.episode_rewards if r > 0)
        player2_wins = sum(1 for r in self.episode_rewards if r < 0)
        self.progress_tracker.log_epoch(self.epoch, policy_loss, policy_loss, player1_wins, player2_wins)

    def save_important_experience(self, state, action, reward, next_state, done):
        """
        ذخیره تجربیات مهم در حافظه بلندمدت.

        Args:
            state: حالت فعلی
            action: حرکت انجام‌شده
            reward: پاداش دریافتی
            next_state: حالت بعدی
            done: آیا بازی تمام شده است
        """
        if reward > self.reward_threshold:
            with open(self.long_term_memory_path, "a") as f:
                experience = {
                    "state": state.cpu().numpy().tolist(),
                    "action": action,
                    "reward": float(reward),
                    "next_state": next_state.cpu().numpy().tolist(),
                    "done": done,
                }
                f.write(json.dumps(experience) + "\n")

    def load_long_term_memory(self):
        """
        بارگذاری تجربیات از حافظه بلندمدت.

        Returns:
            لیست تجربیات
        """
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

    def update_target_network(self):
        """
        به‌روزرسانی شبکه هدف.
        """
        self.episode_count += 1
        if self.episode_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """
        ذخیره مدل و پاداش‌های اپیزود.
        """
        torch.save(self.policy_net.state_dict(), self.model_path)
        torch.save(self.policy_net.state_dict(), self.backup_path)
        with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
            json.dump(self.episode_rewards, f)

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        """
        دریافت متادیتای پیش‌فرض برای این AI.

        Returns:
            دیکشنری شامل default_type و default_description
        """
        return getattr(cls, "AI_METADATA", AI_METADATA)