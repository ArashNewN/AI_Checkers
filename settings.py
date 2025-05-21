from .config import ConfigManager
from .utils import hex_to_rgb
from .checkers_core import log_to_json

# نمونه جهانی ConfigManager
_config_manager = ConfigManager()


class GameSettings:
    def __init__(self, **kwargs):
        # لود تنظیمات عمومی و AI از ConfigManager
        config = _config_manager.load_config()
        ai_config = _config_manager.load_ai_config()

        # تنظیمات عمومی بازی
        self.player_starts = kwargs.get("player_starts", config.get("player_starts"))
        self.use_timer = kwargs.get("use_timer", config.get("use_timer"))
        self.game_time = kwargs.get("game_time", config.get("game_time"))
        self.language = kwargs.get("language", config.get("language"))

        # رنگ‌ها
        try:
            self.player_1_color = hex_to_rgb(kwargs.get("player_1_color", config.get("player_1_color")))
            self.player_2_color = hex_to_rgb(kwargs.get("player_2_color", config.get("player_2_color")))
            self.board_color_1 = hex_to_rgb(kwargs.get("board_color_1", config.get("board_color_1")))
            self.board_color_2 = hex_to_rgb(kwargs.get("board_color_2", config.get("board_color_2")))
        except ValueError as e:
            log_to_json(
                f"Invalid color format in settings: {str(e)}",
                level="ERROR",
                extra_data={"kwargs": kwargs, "config": config}
            )
            raise

        # تنظیمات ظاهری و صوتی
        self.piece_style = kwargs.get("piece_style", config.get("piece_style"))
        self.sound_enabled = kwargs.get("sound_enabled", config.get("sound_enabled"))

        # تنظیمات AI و زمان‌بندی
        self.ai_pause_time = kwargs.get("ai_pause_time", config.get("ai_pause_time"))
        self.game_mode = kwargs.get("game_mode", config.get("game_mode"))
        self.ai_vs_ai_mode = kwargs.get("ai_vs_ai_mode", config.get("ai_vs_ai_mode"))
        self.repeat_hands = kwargs.get("repeat_hands", config.get("repeat_hands"))
        self.pause_between_hands = kwargs.get("pause_between_hands", config.get("pause_between_hands"))

        # نام‌ها
        self.player_1_name = kwargs.get("player_1_name", config.get("player_1_name"))
        self.player_2_name = kwargs.get("player_2_name", config.get("player_2_name"))
        self.al1_name = kwargs.get("al1_name", config.get("al1_name"))
        self.al2_name = kwargs.get("al2_name", config.get("al2_name"))

        # تصاویر
        self.player_1_image = kwargs.get("player_1_image", config.get("player_1_image"))
        self.player_2_image = kwargs.get("player_2_image", config.get("player_2_image"))
        self.al1_image = kwargs.get("al1_image", config.get("al1_image"))
        self.al2_image = kwargs.get("al2_image", config.get("al2_image"))
        self.player_1_piece_image = kwargs.get("player_1_piece_image", config.get("player_1_piece_image"))
        self.player_1_king_image = kwargs.get("player_1_king_image", config.get("player_1_king_image"))
        self.player_2_piece_image = kwargs.get("player_2_piece_image", config.get("player_2_piece_image"))
        self.player_2_king_image = kwargs.get("player_2_king_image", config.get("player_2_king_image"))

        # انواع AI
        self.player_1_ai_type = kwargs.get("player_1_ai_type", config.get("player_1_ai_type"))
        self.player_2_ai_type = kwargs.get("player_2_ai_type", config.get("player_2_ai_type"))

        # توانایی‌های AI
        self.player_1_ability = kwargs.get(
            "player_1_ability",
            ai_config.get("ai_configs", {}).get("player_1", {}).get("ability_level")
        )
        self.player_2_ability = kwargs.get(
            "player_2_ability",
            ai_config.get("ai_configs", {}).get("player_2", {}).get("ability_level")
        )

        # تنظیمات AI
        self.ai_configs = ai_config.get("ai_configs")

        # لاگ‌گیری برای دیباگ
        log_to_json(
            f"GameSettings initialized",
            level="DEBUG",
            extra_data={
                "game_mode": self.game_mode,
                "player_1_ai_type": self.player_1_ai_type,
                "player_2_ai_type": self.player_2_ai_type,
                "ai_configs": self.ai_configs
            }
        )