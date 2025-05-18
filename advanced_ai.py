#advanced_ai.py
from pathlib import Path  # type: ignore
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Optional, Tuple
import sys
from .base_ai import BaseAI
from .config import load_ai_config
from .rewards import RewardCalculator
from .progress_tracker import ProgressTracker, logger

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
    def __init__(self, config):
        super(AdvancedNN, self).__init__()
        nn_params = config["advanced_nn_params"]
        network_params = config["network_params"]
        self.board_size: int = network_params["board_size"]
        self.input_channels: int = nn_params["input_channels"]
        conv1_filters: int = nn_params["conv1_filters"]
        conv1_kernel_size: int = nn_params["conv1_kernel_size"]
        conv1_padding: int = nn_params["conv1_padding"]
        residual_block1_filters: int = nn_params["residual_block1_filters"]
        residual_block2_filters: int = nn_params["residual_block2_filters"]
        conv2_filters: int = nn_params["conv2_filters"]
        self.attention_embed_dim: int = nn_params["attention_embed_dim"]
        attention_num_heads: int = nn_params["attention_num_heads"]
        fc_layer_sizes: list = nn_params["fc_layer_sizes"]
        dropout_rate: float = nn_params["dropout_rate"]

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
        batch_size: int = state.size(0)
        x = state.view(batch_size, self.input_channels, self.board_size, self.board_size)
        x = F.relu(self.conv1(x))
        x = x + self.residual_block1(x)
        x = F.relu(self.conv2(x))
        x = x + self.residual_block2(x)
        attn_input = x.view(batch_size, self.board_size * self.board_size, self.attention_embed_dim)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        residual_block2_filters: int = self.residual_block2[0].in_channels
        attn_output = attn_output.view(batch_size, residual_block2_filters, self.board_size, self.board_size)
        conv_out = attn_output.view(batch_size, -1)
        fc_out = self.fc_layers(conv_out)
        residual_out = self.residual(state.view(batch_size, -1))
        output = fc_out + residual_out
        print(f"Network output shape: {output.shape}")  # Log output shape
        return output

class AdvancedAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, model_name, ai_id):
        super().__init__(game, model_name, ai_id, settings=None)
        logger.info(f"Initialized AdvancedAI for {ai_id} with model {model_name}")

        # ایجاد شبکه‌های عصبی
        self.policy_net = AdvancedNN(self.config).to(self.device)
        self.target_net = AdvancedNN(self.config).to(self.device)
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

    @staticmethod
    def _load_config(config, ai_id):
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
            "update_target_every", "reward_threshold",
            "gradient_clip", "target_update_alpha"
        ]
        for param in required_training_params:
            if param not in player_config["params"]["training_params"]:
                raise KeyError(f"Missing {param} in training_params for {player_key} in ai_config.json")

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

    def act(self, valid_moves: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        logger.info("Executing updated AdvancedAI.act (version 2025-05-18)")
        print("Executing updated AdvancedAI.act (version 2025-05-18)")  # برای کنسول
        # بقیه کد
        print(f"AdvancedAI.act called with valid_moves: {valid_moves}")
        if not valid_moves:
            print("No valid moves in act")
            return None
        if len(valid_moves) == 1:
            move = valid_moves[0]
            print(f"Only one move available: {move}")
            return move  # move is already a tuple

        state = self.get_state(self.game.board.board)
        board_size = self.config["network_params"]["board_size"]
        print(f"State shape: {state.shape}")
        with torch.no_grad():
            q_values = self.policy_net(state)
            print(f"Q-values shape: {q_values.shape}")
            valid_q_values = {}
            for move in valid_moves:
                if not isinstance(move, (tuple, list)) or len(move) != 4:
                    print(f"Invalid move format: {move}")
                    continue
                from_row, from_col, to_row, to_col = move
                index = to_row * board_size + to_col
                print(f"Processing move {move}, calculated index: {index}")
                try:
                    if not (0 <= index < board_size * board_size):
                        print(f"Invalid index {index} for move {move}")
                        continue
                    q_value = q_values[0, index].item()
                    valid_q_values[move] = q_value
                except IndexError as e:
                    print(f"Index error for move {move}: {e}")
                    continue
            print(f"Valid Q-values: {valid_q_values}")
            if valid_q_values:
                best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
                print(f"Best move: {best_move}")
                if best_move not in valid_moves:
                    print(f"Error: Best move {best_move} not in valid_moves")
                    return valid_moves[0]
                return best_move
            print("Warning: No valid Q-values, selecting first valid move")
            fallback_move = valid_moves[0]
            print(f"Fallback move selected: {fallback_move}")
            return fallback_move