import json
import os
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from .base_ai import BaseAI
from .config import load_ai_config
from .rewards import RewardCalculator
from .progress_tracker import ProgressTracker

# متادیتای ماژول
AI_METADATA = {
    "type": "advanced_ai",
    "description": "هوش مصنوعی پیشرفته با شبکه عصبی عمیق و یادگیری تقویتی.",
    "code": "al"
}

project_dir = Path(__file__).parent  # دایرکتوری ماژول (مثل old/<module_folder>)
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

parent_dir = project_dir.parent  # دایرکتوری old
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

class AdvancedNN(nn.Module):
    def __init__(self, config, player_key):
        super(AdvancedNN, self).__init__()
        nn_params = config["advanced_nn_params"]
        network_params = config["network_params"]
        self.board_size = network_params.get("board_size", 8)
        self.input_channels = nn_params.get("input_channels", 3)
        conv1_filters = nn_params.get("conv1_filters", 64)
        conv1_kernel_size = nn_params.get("conv1_kernel_size", 3)
        conv1_padding = nn_params.get("conv1_padding", 1)
        residual_block1_filters = nn_params.get("residual_block1_filters", 64)
        residual_block2_filters = nn_params.get("residual_block2_filters", 128)
        conv2_filters = nn_params.get("conv2_filters", 128)
        self.attention_embed_dim = nn_params.get("attention_embed_dim", 128)
        attention_num_heads = nn_params.get("attention_num_heads", 4)
        fc_layer_sizes = nn_params.get("fc_layer_sizes", [512, 256])
        dropout_rate = nn_params.get("dropout_rate", 0.3)

        self.conv1 = nn.Conv2d(self.input_channels, conv1_filters, kernel_size=conv1_kernel_size, padding=conv1_padding)
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(residual_block1_filters, residual_block1_filters, kernel_size=conv1_kernel_size, padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block1_filters),
            nn.Conv2d(residual_block1_filters, residual_block1_filters, kernel_size=conv1_kernel_size, padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block1_filters),
        )
        self.conv2 = nn.Conv2d(residual_block1_filters, conv2_filters, kernel_size=conv1_kernel_size, padding=conv1_padding)
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(residual_block2_filters, residual_block2_filters, kernel_size=conv1_kernel_size, padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block2_filters),
            nn.Conv2d(residual_block2_filters, residual_block2_filters, kernel_size=conv1_kernel_size, padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block2_filters),
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=attention_num_heads, batch_first=True)

        fc_input_size = residual_block2_filters * self.board_size * self.board_size
        fc_layers = []
        prev_size = fc_input_size
        for size in fc_layer_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, self.board_size * self.board_size))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.residual = nn.Linear(self.board_size * self.board_size * self.input_channels, self.board_size * self.board_size)

    def forward(self, state):
        batch_size = state.size(0)
        x = state.view(batch_size, self.input_channels, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = x + self.residual_block1(x)
        x = F.relu(self.conv2(x))
        x = x + self.residual_block2(x)

        attn_input = x.view(batch_size, self.board_size * self.board_size, self.attention_embed_dim)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.view(batch_size, self.residual_block2[0].in_channels, self.board_size, self.board_size)

        conv_out = attn_output.view(batch_size, -1)
        fc_out = self.fc_layers(conv_out)
        residual_out = self.residual(state.view(batch_size, -1))
        return fc_out + residual_out

class AdvancedAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, model_name, ai_id, config=None):
        super().__init__(game, model_name, ai_id, settings=None)

        # ایجاد شبکه‌های عصبی
        self.policy_net = AdvancedNN(self.config, self.ai_id).to(self.device)
        self.target_net = AdvancedNN(self.config, self.ai_id).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # تنظیم بهینه‌ساز
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config["training_params"]["learning_rate"]
        )

        self.progress_tracker = ProgressTracker(ai_id)
        self.reward_calculator = RewardCalculator(game=game, ai_id=ai_id)

        # بارگذاری مدل در صورت وجود
        if self.model_path.exists():
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _load_config(self, config, ai_id):
        """بارگذاری و اعتبارسنجی تنظیمات از ai_config.json"""
        config_dict = config if config else load_ai_config()
        player_key = "player_1" if ai_id == "ai_1" else "player_2"

        if not config_dict.get("ai_configs", {}).get(player_key):
            raise KeyError(f"No configuration found for {player_key} in ai_config.json")

        player_config = config_dict["ai_configs"][player_key]
        if not player_config.get("params"):
            raise ValueError(f"No parameters defined for {player_key} in ai_config.json")

        # اعتبارسنجی کلیدهای مورد نیاز
        required_params = ["training_params", "network_params", "advanced_nn_params"]
        for param in required_params:
            if param not in player_config["params"]:
                raise KeyError(f"Missing {param} for {player_key} in ai_config.json")

        required_training_params = [
            "memory_size", "batch_size", "learning_rate", "gamma",
            "epsilon_start", "epsilon_end", "epsilon_decay",
            "update_target_every", "reward_threshold"
        ]
        for param in required_training_params:
            if param not in player_config["params"]["training_params"]:
                raise KeyError(f"Missing {param} in training_params for {player_key} in ai_config.json")

        # افزودن مقادیر پیش‌فرض برای پارامترهای اختیاری
        player_config["params"]["training_params"]["gradient_clip"] = player_config["params"]["training_params"].get("gradient_clip", 1.0)
        player_config["params"]["training_params"]["target_update_alpha"] = player_config["params"]["training_params"].get("target_update_alpha", 0.01)

        # تنظیم مسیر پیش‌فرض نسبی برای model_dir
        model_dir = config_dict.get("model_dir", str(project_dir / "models"))

        if "ability_level" not in player_config:
            raise KeyError(f"Missing ability_level for {player_key} in ai_config.json")

        # بازگشت تنظیمات به صورت دیکشنری
        return {
            "model_dir": model_dir,
            "ability_level": player_config["ability_level"],
            "training_params": player_config["params"]["training_params"],
            "network_params": player_config["params"]["network_params"],
            "advanced_nn_params": player_config["params"]["advanced_nn_params"],
            "reward_weights": player_config["params"].get("reward_weights", {}),
            "mcts_params": player_config["params"].get("mcts_params", {}),
            "end_game_rewards": player_config["params"].get("end_game_rewards", {})
        }

    def act(self, valid_moves):
        """
        انتخاب حرکت از بین حرکت‌های معتبر با استفاده از شبکه عصبی.

        Args:
            valid_moves: دیکشنری حرکت‌های معتبر

        Returns:
            حرکت انتخاب‌شده
        """
        print(f"AdvancedAI.act called with valid_moves: {valid_moves}")
        if not valid_moves:
            print("No valid moves in act")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            print(f"Only one move available: {move}")
            return move

        state = self.get_state(self.game.board.board)
        board_size = self.config["network_params"].get("board_size", 8)
        print(f"State shape: {state.shape}")
        with torch.no_grad():
            q_values = self.policy_net(state)
            print(f"Q-values shape: {q_values.shape}")
            valid_q_values = {}
            for (position, move), skipped in valid_moves.items():
                index = move[0] * board_size + move[1]
                try:
                    q_value = q_values[0, index].item()
                    valid_q_values[(position, move)] = q_value
                except IndexError as e:
                    print(f"Index error for move {move}: {e}")
                    continue
            print(f"Valid Q-values: {valid_q_values}")
            if valid_q_values:
                best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
                print(f"Best move: {best_move}")
                return best_move
            print("Warning: No valid Q-values, selecting first valid move")
            return list(valid_moves.keys())[0]