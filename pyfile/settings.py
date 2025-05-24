from .config import ConfigManager, log_to_json
from .utils import hex_to_rgb


# نمونه جهانی ConfigManager
_config_manager = ConfigManager()


class GameSettings:
    def __init__(self, **kwargs):
        # لود تنظیمات عمومی و AI از ConfigManager
        self.ai_1_image = None
        self.ai_2_image = None
        self.ai_2_name = None
        self.ai_1_name = None
        config = _config_manager.load_config()
        ai_config = _config_manager.load_ai_config()

        # تنظیمات عمومی بازی
        self.player_starts = kwargs.get("player_starts", config.get("player_starts", True))
        self.use_timer = kwargs.get("use_timer", config.get("use_timer", False))
        self.game_time = kwargs.get("game_time", config.get("game_time", 5))
        self.language = kwargs.get("language", config.get("language", "en"))

        # رنگ‌ها
        try:
            self.player_1_color = hex_to_rgb(kwargs.get("player_1_color", config.get("player_1_color", "#FF0000")))
            self.player_2_color = hex_to_rgb(kwargs.get("player_2_color", config.get("player_2_color", "#0000FF")))
            self.board_color_1 = hex_to_rgb(kwargs.get("board_color_1", config.get("board_color_1", "#FFFFFF")))
            self.board_color_2 = hex_to_rgb(kwargs.get("board_color_2", config.get("board_color_2", "#000000")))
        except ValueError as e:
            log_to_json(
                f"Invalid color format in settings: {str(e)}",
                level="ERROR",
                extra_data={"kwargs": kwargs, "config": config}
            )
            raise

        # تنظیمات ظاهری و صوتی
        self.piece_style = kwargs.get("piece_style", config.get("piece_style", "circle"))
        self.sound_enabled = kwargs.get("sound_enabled", config.get("sound_enabled", True))

        # تنظیمات AI و زمان‌بندی
        self.ai_pause_time = kwargs.get("ai_pause_time", config.get("ai_pause_time", 100))
        self.game_mode = kwargs.get("game_mode", config.get("game_mode", "human_vs_human"))
        self.ai_vs_ai_mode = kwargs.get("ai_vs_ai_mode", config.get("ai_vs_ai_mode", "only_once"))
        self.repeat_hands = kwargs.get("repeat_hands", config.get("repeat_hands", 1))
        self.pause_between_hands = kwargs.get("pause_between_hands", config.get("pause_between_hands", 0))

        # نام‌ها
        self.player_1_name = kwargs.get("player_1_name", config.get("player_1_name", "Player 1"))
        self.player_2_name = kwargs.get("player_2_name", config.get("player_2_name", "Player 2"))

        # تصاویر
        self.player_1_image = kwargs.get("player_1_image", config.get("player_1_image", ""))
        self.player_2_image = kwargs.get("player_2_image", config.get("player_2_image", ""))
        self.player_1_piece_image = kwargs.get("player_1_piece_image", config.get("player_1_piece_image", ""))
        self.player_1_king_image = kwargs.get("player_1_king_image", config.get("player_1_king_image", ""))
        self.player_2_piece_image = kwargs.get("player_2_piece_image", config.get("player_2_piece_image", ""))
        self.player_2_king_image = kwargs.get("player_2_king_image", config.get("player_2_king_image", ""))

        # انواع AI
        self.player_1_ai_type = kwargs.get(
            "player_1_ai_type",
            ai_config.get("ai_configs", {}).get("player_1", {}).get("ai_type", "none")
        )
        self.player_2_ai_type = kwargs.get(
            "player_2_ai_type",
            ai_config.get("ai_configs", {}).get("player_2", {}).get("ai_type", "none")
        )

        # توانایی‌های AI
        self.player_1_ability = kwargs.get(
            "player_1_ability",
            ai_config.get("ai_configs", {}).get("player_1", {}).get("ability_level", 5)
        )
        self.player_2_ability = kwargs.get(
            "player_2_ability",
            ai_config.get("ai_configs", {}).get("player_2", {}).get("ability_level", 5)
        )

        # تنظیمات AI
        self.ai_configs = ai_config.get("ai_configs", {
            "player_1": {"ai_type": "none", "ai_code": None, "ability_level": 5, "params": {}},
            "player_2": {"ai_type": "none", "ai_code": None, "ability_level": 5, "params": {}}
        })

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