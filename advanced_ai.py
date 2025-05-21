# advanced_ai.py

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base_ai import BaseAI
from .config import ConfigManager, DEFAULT_AI_PARAMS
from .progress_tracker import ProgressTracker, logger
from .rewards import RewardCalculator

# نمونه ConfigManager برای مدیریت مرکزی تنظیمات
_config_manager = ConfigManager()

AI_METADATA = {
    "type": "advanced_ai",
    "description": "هوش مصنوعی پیشرفته با شبکه عصبی عمیق و یادگیری تقویتی.",
    "code": "al"
}

# تنظیم مسیرهای سیستمی
project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

parent_dir = project_dir.parent
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
            nn.Conv2d(residual_block1_filters, residual_block1_filters, kernel_size=conv1_kernel_size,
                      padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block1_filters),
            nn.Conv2d(residual_block1_filters, residual_block1_filters, kernel_size=conv1_kernel_size,
                      padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block1_filters),
        )
        self.conv2 = nn.Conv2d(residual_block1_filters, conv2_filters, kernel_size=conv1_kernel_size,
                               padding=conv1_padding)
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(residual_block2_filters, residual_block2_filters, kernel_size=conv1_kernel_size,
                      padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block2_filters),
            nn.Conv2d(residual_block2_filters, residual_block2_filters, kernel_size=conv1_kernel_size,
                      padding=conv1_padding),
            nn.ReLU(),
            nn.BatchNorm2d(residual_block2_filters),
        )

        self.attention = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=attention_num_heads,
                                               batch_first=True)

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

        self.residual = nn.Linear(self.board_size * self.board_size * self.input_channels,
                                  self.board_size * self.board_size)

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
        logger.debug(f"Network output shape: {output.shape}")
        return output

class AdvancedAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, model_name, ai_id, config=None):
        # بارگذاری تنظیمات از ConfigManager
        config = self._load_config(config, ai_id)
        super().__init__(game, model_name, ai_id, settings=config)
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

        # ذخیره تنظیمات در فایل اختصاصی (al_config.json)
        self._save_specific_config()

    def _save_specific_config(self):
        """ذخیره تنظیمات در فایل اختصاصی AI (مثل al_config.json) با استفاده از ConfigManager"""
        try:
            config_path = _config_manager.get_ai_specific_config_path(self.metadata['code'])
            config_to_save = {
                "player_1" if self.ai_id == "ai_1" else "player_2": self.config
            }
            _config_manager.save_ai_specific_config(self.metadata['code'], config_to_save)
            logger.info(f"Saved AI specific config to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save AI specific config to {config_path}: {str(e)}")

    def _load_config(self, config, ai_id):
        """بارگذاری و اعتبارسنجی تنظیمات از ConfigManager"""
        try:
            config_dict = config if config else _config_manager.load_ai_config()
            player_key = "player_1" if ai_id == "ai_1" else "player_2"

            player_config = config_dict.get("ai_configs", {}).get(player_key, {})
            if not isinstance(player_config, dict):
                logger.warning(f"Invalid configuration format for {player_key}, using default")
                return DEFAULT_AI_PARAMS.copy()

            params = player_config.get("params", {})
            if not isinstance(params, dict):
                logger.warning(f"No valid parameters for {player_key}, using default")
                return DEFAULT_AI_PARAMS.copy()

            # پر کردن پارامترهای غایب با DEFAULT_AI_PARAMS
            required_params = ["training_params", "network_params", "advanced_nn_params"]
            for param in required_params:
                if param not in params:
                    params[param] = DEFAULT_AI_PARAMS.get(param, {}).copy()
                else:
                    for key, value in DEFAULT_AI_PARAMS.get(param, {}).items():
                        if key not in params[param]:
                            params[param][key] = value

            required_training_params = [
                "memory_size", "batch_size", "learning_rate", "gamma",
                "epsilon_start", "epsilon_end", "epsilon_decay",
                "update_target_every", "reward_threshold",
                "gradient_clip", "target_update_alpha"
            ]
            for param in required_training_params:
                if param not in params["training_params"]:
                    params["training_params"][param] = DEFAULT_AI_PARAMS["training_params"].get(param)

            # تنظیم مسیر model_dir از ai_config.json
            model_dir = config_dict.get("model_dir", "models")

            return {
                "model_dir": model_dir,
                "ability_level": player_config.get("ability_level", DEFAULT_AI_PARAMS["ability_level"]),
                "training_params": params["training_params"],
                "network_params": params["network_params"],
                "advanced_nn_params": params["advanced_nn_params"],
                "reward_weights": params.get("reward_weights", DEFAULT_AI_PARAMS["reward_weights"].copy()),
                "mcts_params": params.get("mcts_params", DEFAULT_AI_PARAMS["mcts_params"].copy()),
                "end_game_rewards": params.get("end_game_rewards", DEFAULT_AI_PARAMS["end_game_rewards"].copy())
            }
        except Exception as e:
            logger.error(f"Error loading config for {ai_id}: {str(e)}, using default")
            return DEFAULT_AI_PARAMS.copy()

    def act(self, valid_moves: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        logger.info("Executing AdvancedAI.act (version 2025-05-21)")
        logger.debug(f"Valid moves: {valid_moves}")
        if not valid_moves:
            logger.warning("No valid moves available")
            return None
        if len(valid_moves) == 1:
            move = valid_moves[0]
            logger.info(f"Only one move available: {move}")
            return move

        state = self.get_state(self.game.board.board)
        board_size = self.config["network_params"]["board_size"]
        logger.debug(f"State shape: {state.shape}")
        try:
            with torch.no_grad():
                q_values = self.policy_net(state)
                logger.debug(f"Q-values shape: {q_values.shape}")
                valid_q_values = {}
                max_index = board_size * board_size * board_size * board_size
                for move in valid_moves:
                    if not isinstance(move, tuple) or len(move) != 4:
                        logger.error(f"Invalid move format: {move}")
                        continue
                    from_row, from_col, to_row, to_col = move
                    index = (from_row * board_size + from_col) * (board_size * board_size) + (
                                to_row * board_size + to_col)
                    logger.debug(f"Processing move {move}, calculated index: {index}")
                    if not (0 <= index < max_index):
                        logger.error(f"Invalid index {index} for move {move}")
                        continue
                    try:
                        q_value = q_values[0, index].item()
                        valid_q_values[move] = q_value
                    except IndexError as e:
                        logger.error(f"Index error for move {move}: {e}")
                        continue
                logger.debug(f"Valid Q-values: {valid_q_values}")
                if valid_q_values:
                    best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
                    logger.info(f"Best move selected: {best_move}")
                    return best_move
                logger.warning("No valid Q-values computed, selecting first valid move")
                return valid_moves[0]
        except Exception as e:
            logger.error(f"Error in act: {str(e)}")
            logger.info("Selecting first valid move as fallback")
            return valid_moves[0]