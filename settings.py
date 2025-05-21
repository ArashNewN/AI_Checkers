from .config import load_config
from .utils import hex_to_rgb

class GameSettings:
    def __init__(self):
        config = load_config()
        self.player_starts = config["player_starts"]
        self.use_timer = config["use_timer"]
        self.game_time = config["game_time"]
        self.language = config["language"]
        self.player_1_color = hex_to_rgb(config["player_1_color"])
        self.player_2_color = hex_to_rgb(config["player_2_color"])
        self.board_color_1 = hex_to_rgb(config["board_color_1"])
        self.board_color_2 = hex_to_rgb(config["board_color_2"])
        self.piece_style = config["piece_style"]
        self.sound_enabled = config["sound_enabled"]
        self.ai_pause_time = config["ai_pause_time"]
        self.az1_ability = config["az1_ability"]
        self.az2_ability = config["az2_ability"]
        self.ai_1_ability = config["ai_1_ability"]
        self.ai_2_ability = config["ai_2_ability"]
        self.game_mode = config["game_mode"]
        self.ai_vs_ai_mode = config["ai_vs_ai_mode"]
        self.repeat_hands = config["repeat_hands"]
        self.player_1_name = config["player_1_name"]
        self.player_2_name = config["player_2_name"]
        self.ai_1_name = config["ai_1_name"]
        self.ai_2_name = config["ai_2_name"]
        self.player_1_image = config["player_1_image"]
        self.player_2_image = config["player_2_image"]
        self.ai_1_image = config["ai_1_image"]
        self.ai_2_image = config["ai_2_image"]
        self.player_1_piece_image = config["player_1_piece_image"]
        self.player_1_king_image = config["player_1_king_image"]
        self.player_2_piece_image = config["player_2_piece_image"]
        self.player_2_king_image = config["player_2_king_image"]
        self.pause_between_hands = config["pause_between_hands"]
        self.player_1_ai_type = config["player_1_ai_type"]
        self.player_2_ai_type = config["player_2_ai_type"]