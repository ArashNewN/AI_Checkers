from .config import load_config
from .utils import hex_to_rgb


class GameSettings:
    def __init__(self):
        config = load_config()
        self.player_starts = config.get("player_starts", True)
        self.use_timer = config.get("use_timer", True)
        self.game_time = config.get("game_time", 5)
        self.language = config.get("language", "en")
        self.player_1_color = hex_to_rgb(config.get("player_1_color", "#ff0000"))
        self.player_2_color = hex_to_rgb(config.get("player_2_color", "#0000ff"))
        self.board_color_1 = hex_to_rgb(config.get("board_color_1", "#ffffff"))
        self.board_color_2 = hex_to_rgb(config.get("board_color_2", "#8b4513"))
        self.piece_style = config.get("piece_style", "circle")
        self.sound_enabled = config.get("sound_enabled", False)
        self.ai_pause_time = config.get("ai_pause_time", 20)
        self.game_mode = config.get("game_mode", "human_vs_human")
        self.ai_vs_ai_mode = config.get("ai_vs_ai_mode", "only_once")
        self.repeat_hands = config.get("repeat_hands", 10)
        self.player_1_name = config.get("player_1_name", "Player 1")
        self.player_2_name = config.get("player_2_name", "Player 2")
        self.al1_name = config.get("al1_name", "AI 1")
        self.al2_name = config.get("al2_name", "AI 2")
        self.player_1_image = config.get("player_1_image", "")
        self.player_2_image = config.get("player_2_image", "")
        self.al1_image = config.get("al1_image", "")
        self.al2_image = config.get("al2_image", "")
        self.player_1_piece_image = config.get("player_1_piece_image", "")
        self.player_1_king_image = config.get("player_1_king_image", "")
        self.player_2_piece_image = config.get("player_2_piece_image", "")
        self.player_2_king_image = config.get("player_2_king_image", "")
        self.pause_between_hands = config.get("pause_between_hands", 1000)
        self.player_1_ai_type = config.get("player_1_ai_type", "none")
        self.player_2_ai_type = config.get("player_2_ai_type", "none")
        # تنظیمات عمومی برای AIها
        self.player_1_ability = config.get("ai_configs", {}).get("player_1", {}).get("ability_level", 5)
        self.player_2_ability = config.get("ai_configs", {}).get("player_2", {}).get("ability_level", 5)
        # اضافه کردن ai_configs
        self.ai_configs = config.get("ai_configs", {
            "player_1": {
                "ai_type": "none",
                "ability_level": 5,
                "training_params": {},
                "reward_weights": {
                    "piece_difference": 1.0,
                    "king_bonus": 2.0,
                    "position_bonus": 0.1,
                    "capture_bonus": 1.0,
                    "multi_jump_bonus": 2.0,
                    "king_capture_bonus": 3.0,
                    "mobility_bonus": 0.1,
                    "safety_penalty": -0.5
                }
            },
            "player_2": {
                "ai_type": "none",
                "ability_level": 5,
                "training_params": {},
                "reward_weights": {
                    "piece_difference": 1.0,
                    "king_bonus": 2.0,
                    "position_bonus": 0.1,
                    "capture_bonus": 1.0,
                    "multi_jump_bonus": 2.0,
                    "king_capture_bonus": 3.0,
                    "mobility_bonus": 0.1,
                    "safety_penalty": -0.5
                }
            }
        })
