import logging
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .base_ai import BaseAI
from pyfile.config import load_ai_config
from .rewards import RewardCalculator
from .progress_tracker import ProgressTracker
from pyfile.checkers_game import CheckersGame

AI_METADATA = {
    "type": "alphazero_ai",
    "description": "هوش مصنوعی پیشرفته مبتنی بر الگوریتم AlphaZero با MCTS.",
    "code": "az"
}

project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

parent_dir = project_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

log_dir = os.path.dirname(os.path.dirname(__file__))
log_file = os.path.join(log_dir, 'alphazero_ai.log')
logging.basicConfig(
    level=logging.DEBUG,
    format=' %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)


# به جای import واقعی، یک کلاس موقت تعریف می‌کنیم
class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, x):
        return torch.zeros((x.size(0), 64))  # خروجی موقت

# به جای MCTS، یک کلاس موقت تعریف می‌کنیم
class MCTS:
    def __init__(self, ai, game, model):
        self.ai = ai
        self.game = game
        self.model = model


    def search(state, valid_moves):
        # خروجی موقت برای تست
        move_probs = [1.0 / len(valid_moves) for _ in valid_moves]
        value = 0.0
        return move_probs, value


class AlphaZeroAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, model_name, ai_id, config=None):
        super().__init__(game, model_name, ai_id, settings=None)

        self.color = "white" if self.player_number == 1 else "black"

        # شبکه‌های عصبی
        self.policy_net = AlphaZeroNet().to(self.device)
        self.target_net = AlphaZeroNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config["training_params"]["learning_rate"],
            weight_decay=self.config["training_params"]["weight_decay"]
        )

        self.progress_tracker = ProgressTracker(ai_id)
        self.mcts = MCTS(self, CheckersGame(), self.policy_net)
        self.reward_calculator = RewardCalculator(game=self.game, ai_id=ai_id)

        # بارگذاری مدل‌های قبلی
        if self.model_path.exists():
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

        logger.debug(f"Initialized AlphaZeroAI with ai_id={self.ai_id}, ability={self.ability}, color={self.color}")

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
        required_params = ["training_params", "advanced_nn_params"]
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
        player_config["params"]["training_params"]["gradient_clip"] = player_config["params"]["training_params"].get(
            "gradient_clip", 1.0)
        player_config["params"]["training_params"]["target_update_alpha"] = player_config["params"]["training_params"].get(
            "target_update_alpha", 0.01)
        player_config["params"]["training_params"]["weight_decay"] = player_config["params"]["training_params"].get(
            "weight_decay", 0.01)

        # تنظیم network_params پیش‌فرض اگر غایب باشد
        if "network_params" not in player_config["params"]:
            player_config["params"]["network_params"] = {
                "board_size": 8
            }

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
        انتخاب حرکت از بین حرکت‌های معتبر با استفاده از MCTS.

        Args:
            valid_moves: دیکشنری حرکت‌های معتبر

        Returns:
            حرکت انتخاب‌شده
        """
        print(f"AlphaZeroAI.act called with valid_moves: {valid_moves}")
        if not valid_moves:
            print("No valid moves in act")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            print(f"Only one move available: {move}")
            return move

        state = self.get_state(self.game.board.board)
        print(f"State shape: {state.shape}")
        move_probs, _ = self.mcts.search(state, valid_moves)

        # انتخاب حرکت با بالاترین احتمال
        valid_move_keys = list(valid_moves.keys())
        move_probs_dict = {move: prob for move, prob in zip(valid_move_keys, move_probs)}
        print(f"Move probabilities: {move_probs_dict}")

        if move_probs_dict:
            best_move = max(move_probs_dict.items(), key=lambda x: x[1])[0]
            print(f"Best move: {best_move}")
            return best_move
        else:
            print("Warning: No valid move probabilities, selecting first valid move")
            return valid_move_keys[0]

    def _row_col_to_idx(self, row, col):
        if not (0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1):
            logger.warning(f"Invalid row,col: ({row}, {col})")
            return None
        idx = ((row // 2) * 4) + ((col // 2) if row % 2 == 0 else (col // 2))
        if not (0 <= idx < 32):
            logger.warning(f"Invalid idx from row,col ({row}, {col}): {idx}")
            return None
        return idx