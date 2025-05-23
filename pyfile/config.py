import json
import os
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
import importlib
from typing import Dict, Union, Optional

try:
    from .constants import LANGUAGES
except ImportError as e:
    logging.error(f"Failed to import LANGUAGES from constants: {e}")
    LANGUAGES = {"en": {}, "fa": {}}  # فال‌بک پیش‌فرض

logger = logging.getLogger(__name__)

# Default ability levels mapping with string keys
DEFAULT_ABILITY_LEVELS: Dict[str, str] = {
    "1": "very_weak",
    "3": "weak",
    "5": "medium",
    "7": "strong",
    "9": "very_strong"
}

# Default AI parameters
DEFAULT_AI_PARAMS = {
    "training_params": {
        "memory_size": 10000,
        "batch_size": 128,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.999,
        "update_target_every": 100,
        "reward_threshold": 0.5
    },
    "reward_weights": {
        "piece_difference": 1.0,
        "king_bonus": 2.0,
        "position_bonus": 0.1,
        "capture_bonus": 1.0,
        "multi_jump_bonus": 2.0,
        "king_capture_bonus": 3.0,
        "mobility_bonus": 0.1,
        "safety_penalty": -0.5
    },
    "mcts_params": {
        "c_puct": 1.0,
        "num_simulations": 200,
        "max_cache_size": 10000,
        "num_processes": 4,
        "cache_file": "state_cache.json.gz",
        "cache_save_interval": 100
    },
    "network_params": {
        "input_channels": 4,
        "num_filters": 64,
        "num_blocks": 8,
        "board_size": 8,
        "num_actions": 1024,
        "dropout_rate": 0.3
    },
    "advanced_nn_params": {
        "input_channels": 3,
        "conv1_filters": 64,
        "conv1_kernel_size": 3,
        "conv1_padding": 1,
        "residual_block1_filters": 64,
        "residual_block2_filters": 128,
        "conv2_filters": 128,
        "attention_embed_dim": 128,
        "attention_num_heads": 4,
        "fc_layer_sizes": [512, 256],
        "dropout_rate": 0.3
    },
    "end_game_rewards": {
        "win_no_timeout": 100,
        "win_timeout": 0,
        "draw": -50,
        "loss": -100
    }
}

# LazyLoader class for lazy-loading modules
class LazyLoader:
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None
        self._class = None

    def load(self):
        if self._module is None:
            try:
                self._module = importlib.import_module(self.module_name)
                if not hasattr(self._module, self.class_name):
                    raise AttributeError(f"Class '{self.class_name}' not found in module {self.module_name}")
                self._class = getattr(self._module, self.class_name)
                logger.debug(f"Lazy-loaded module {self.module_name}, class {self.class_name}")
            except Exception as load_error:
                logger.error(f"Error lazy-loading module {self.module_name}: {load_error}")
                raise
        return self._class

# ConfigManager class for centralized configuration management
class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._config_cache = None
        self._ai_config_cache = None
        self._ai_specific_config_cache = {}
        self._ai_module_cache = {}
        self.project_root = None
        self._initialized = True

        # مسیرهای پویا برای پوشه‌ها
        self.config_dir = self.get_project_root() / "configs"
        self.log_dir = self.get_project_root() / "logs"
        self.pth_dir = self.get_project_root() / "pth"
        self.assets_dir = self.get_project_root() / "assets"

        # ایجاد پوشه‌ها در صورت عدم وجود
        for directory in [self.config_dir, self.log_dir, self.pth_dir, self.assets_dir, self.config_dir / "ai"]:
            directory.mkdir(parents=True, exist_ok=True)

        # تنظیم لاگ‌گیری
        logging.basicConfig(
            level=logging.DEBUG,
            filename=self.log_dir / "app.log",
            encoding="utf-8",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def get_project_root(self) -> Path:
        if self.project_root is None:
            # بررسی متغیر محیطی PROJECT_ROOT
            env_project_root = os.getenv("PROJECT_ROOT")
            if env_project_root and Path(env_project_root).exists():
                self.project_root = Path(env_project_root).resolve()
                logger.debug(f"Project root set from PROJECT_ROOT env: {self.project_root}")
                return self.project_root

            # شروع از مسیر فایل فعلی
            current_path = Path(__file__).resolve().parent
            max_depth = 15
            depth = 0
            while depth < max_depth:
                configs_dir = current_path / "configs"
                modules_dir = current_path / "modules"
                logs_dir = current_path / "logs"
                if any([configs_dir.exists(), modules_dir.exists(), logs_dir.exists()]):
                    logger.debug(f"Project root found at: {current_path}")
                    self.project_root = current_path
                    for directory in [configs_dir, modules_dir, logs_dir]:
                        directory.mkdir(parents=True, exist_ok=True)
                    return self.project_root
                current_path = current_path.parent
                depth += 1

            logger.warning("Could not find project root, using default path")
            self.project_root = Path(__file__).resolve().parent.parent
            for directory in [self.project_root / "configs", self.project_root / "modules", self.project_root / "logs"]:
                directory.mkdir(parents=True, exist_ok=True)
        return self.project_root

    @staticmethod
    def log_to_json(message: str, level: str = "DEBUG", extra_data: Optional[Dict] = None):
        def convert_to_serializable(data):
            if isinstance(data, dict):
                return {str(k) if isinstance(k, (tuple, list)) else k: convert_to_serializable(v) for k, v in data.items()}
            elif isinstance(data, (np.integer, np.floating)):
                return data.item()
            elif isinstance(data, (list, tuple)):
                return [convert_to_serializable(item) for item in data]
            elif isinstance(data, Path):
                return str(data)
            return data

        try:
            extra_data = convert_to_serializable(extra_data) if extra_data else {}
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                "extra_data": extra_data
            }
            log_path = ConfigManager().log_dir / "json_logs.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write("\n")
            logger.log(getattr(logging, level), f"{message} | Extra: {extra_data}")
        except Exception as log_error:
            logger.error(f"Error logging to JSON: {log_error}")

    def get_stats_path(self) -> Path:
        return self.config_dir / "stats.json"

    def get_config_path(self) -> Path:
        return self.config_dir / "config.json"

    def get_ai_config_path(self) -> Path:
        return self.config_dir / "ai_config.json"

    def get_ai_specific_config_path(self, ai_code: str) -> Path:
        return self.config_dir / "ai" / f"{ai_code}_config.json"

    def load_config(self) -> Dict:
        if self._config_cache is not None:
            logger.debug("Returning cached config")
            return self._config_cache

        default_config = {
            "square_size": 80,
            "board_size": 8,
            "border_thickness": 7,
            "menu_height": 30,
            "window_width": 940,
            "board_width": 640,
            "window_height": 720,
            "panel_width": 300,
            "button_spacing_from_bottom": 40,
            "animation_frames": 100,
            "player_image_size": 75,
            "settings_window_width": 500,
            "settings_window_height": 750,
            "min_window_width": 300,
            "min_window_height": 200,
            "min_game_window_width": 600,
            "min_game_window_height": 400,
            "progress_window_width": 600,
            "progress_window_height": 400,
            "help_window_width": 300,
            "help_window_height": 200,
            "about_window_width": 300,
            "about_window_height": 200,
            "advanced_config_window_width": 500,
            "advanced_config_window_height": 600,
            "game_version": "1.0",
            "hint_enabled_p1_default": False,
            "hint_enabled_p2_default": False,
            "hint_circle_color": [255, 165, 0],
            "hint_circle_radius": 10,
            "hint_blink_interval": 500,
            "hint_button_width": 120,
            "hint_button_height": 40,
            "hint_button_spacing": 10,
            "hint_button_y_offset": 10,
            "undo_button_width": 120,
            "undo_button_height": 40,
            "redo_button_width": 120,
            "redo_button_height": 40,
            "undo_redo_button_spacing": 10,
            "undo_redo_y_offset": 10,
            "black": [0, 0, 0],
            "white": [255, 255, 255],
            "red": [255, 0, 0],
            "gray": [128, 128, 128],
            "blue": [0, 0, 255],
            "sky_blue": [135, 206, 235],
            "light_gray": [211, 211, 211],
            "light_green": [144, 238, 144],
            "piece_style": "circle",
            "sound_enabled": False,
            "ai_pause_time": 20,
            "game_mode": "human_vs_human",
            "ai_vs_ai_mode": "only_once",
            "repeat_hands": 100,
            "player_1_name": "Player 1",
            "player_2_name": "Player 2",
            "al1_name": "AI 1",
            "al2_name": "AI 2",
            "player_1_image": "",
            "player_2_image": "",
            "al1_image": "",
            "al2_image": "",
            "player_1_piece_image": str(self.assets_dir / "pieces" / "red_piece.png"),
            "player_1_king_image": str(self.assets_dir / "pieces" / "red_king.png"),
            "player_2_piece_image": str(self.assets_dir / "pieces" / "blue_piece.png"),
            "player_2_king_image": str(self.assets_dir / "pieces" / "blue_king.png"),
            "pause_between_hands": 1000,
            "player_1_ai_type": "none",
            "player_2_ai_type": "none",
            "use_timer": False,
            "game_time": 5,
            "language": "en",
            "player_1_color": "#ff0000",
            "player_2_color": "#0000ff",
            "board_color_1": "#ffffff",
            "board_color_2": "#8b4513",
            "assets_dir": str(self.assets_dir),
            "max_no_capture_moves": 40,
            "max_uniform_moves": 5,
            "max_total_moves": 40,
            "logging_level": "DEBUG"
        }
        config_path = self.get_config_path()
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if not isinstance(config, dict):
                        raise ValueError("Config file must contain a dictionary")
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_key not in config[key]:
                                    config[key][sub_key] = sub_value
            else:
                logger.debug(f"Config file not found at {config_path}, creating with default config")
                config = default_config
                self.save_config(config)
            self._config_cache = config
            logger.debug("Config cached successfully")
        except (json.JSONDecodeError, ValueError, Exception) as config_error:
            logger.error(f"Error loading config from {config_path}: {config_error}, using default config")
            config = default_config
            self._config_cache = config
            self.save_config(config)
        return self._config_cache

    def save_config(self, config: Dict):
        config_path = self.get_config_path()
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.debug(f"Config saved to {config_path}")
            self._config_cache = config
        except Exception as save_config_error:
            logger.error(f"Error saving config to {config_path}: {save_config_error}")

    def load_ai_config(self) -> Dict[str, Union[Dict, Dict[str, str], Dict[str, Dict]]]:
        if self._ai_config_cache is not None:
            logger.debug("Returning cached AI config")
            return self._ai_config_cache

        default_ai_config = {
            "ai_types": {},
            "ability_levels": DEFAULT_ABILITY_LEVELS,
            "ai_configs": {
                "player_1": {
                    "ai_type": "none",
                    "ai_code": None,
                    "ability_level": 5,
                    "params": DEFAULT_AI_PARAMS.copy()
                },
                "player_2": {
                    "ai_type": "none",
                    "ai_code": None,
                    "ability_level": 5,
                    "params": DEFAULT_AI_PARAMS.copy()
                }
            }
        }

        ai_config_path = self.get_ai_config_path()
        ai_config = default_ai_config
        config_changed = False

        try:
            if ai_config_path.exists():
                with open(ai_config_path, "r", encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    if not isinstance(loaded_config, dict):
                        raise ValueError("AI config file must contain a dictionary")
                    ai_config = loaded_config
                    for key, value in default_ai_config.items():
                        if key not in ai_config:
                            ai_config[key] = value
                            config_changed = True
                        elif isinstance(value, dict) and key != "ability_levels":
                            for sub_key, sub_value in value.items():
                                if sub_key not in ai_config[key]:
                                    ai_config[key][sub_key] = sub_value
                                    config_changed = True
                    logger.debug(f"Loaded AI config from {ai_config_path}")
            else:
                logger.debug(f"AI config file not found at {ai_config_path}, creating with default config")
                config_changed = True
        except (json.JSONDecodeError, ValueError, Exception) as ai_config_error:
            logger.error(f"Error loading AI config from {ai_config_path}: {ai_config_error}, using default AI config")
            ai_config = default_ai_config
            config_changed = True

        used_codes = set()
        valid_ai_types = {}
        for ai_type, ai_info in ai_config["ai_types"].items():
            ai_info: Dict[str, str]
            if not isinstance(ai_info, dict):
                logger.warning(f"Invalid AI info for {ai_type}, skipping")
                continue
            code = ai_info.get("code", "")
            if not code or len(code) != 2:
                logger.warning(f"Invalid or missing code for AI {ai_type}, skipping")
                continue
            if code in used_codes:
                logger.warning(f"Duplicate AI code detected: {code}, skipping")
                continue
            used_codes.add(code)
            valid_ai_types[ai_type] = ai_info
            logger.debug(f"Validated AI: type {ai_type}, code {code}")

        if ai_config["ai_types"] != valid_ai_types:
            ai_config["ai_types"] = valid_ai_types
            config_changed = True

        valid_ai_types = {}
        for ai_type, ai_info in ai_config["ai_types"].items():
            ai_info: Dict[str, str]
            module_name = ai_info.get("module", "")
            class_name = ai_info.get("class", "")
            module_path = self.get_project_root() / "modules" / f"{module_name}.py"

            logger.debug(f"Checking module: {module_path} for AI type {ai_type}")
            if module_path.exists():
                module_import_name = f"modules.{module_name}"
                self._ai_module_cache[ai_type] = LazyLoader(module_import_name, class_name)
                valid_ai_types[ai_type] = ai_info
                logger.debug(f"AI type {ai_type} prepared for lazy loading: module {module_import_name}, class {class_name}")
                ai_code = ai_info.get("code", "")
                if ai_code:
                    ai_specific_config_path = self.get_ai_specific_config_path(ai_code)
                    if not ai_specific_config_path.exists():
                        self.save_ai_specific_config(ai_code, {
                            "player_1": DEFAULT_AI_PARAMS.copy(),
                            "player_2": DEFAULT_AI_PARAMS.copy()
                        })
                        config_changed = True
            else:
                logger.warning(f"AI type {ai_type} ignored: module file {module_path} not found")

        if ai_config["ai_types"] != valid_ai_types:
            ai_config["ai_types"] = valid_ai_types
            config_changed = True

        for player in ["player_1", "player_2"]:
            player_config: Dict[str, Union[str, None, int, Dict]] = ai_config["ai_configs"].get(player, {})
            if not isinstance(player_config, dict):
                player_config = {
                    "ai_type": "none",
                    "ai_code": None,
                    "ability_level": 5,
                    "params": DEFAULT_AI_PARAMS.copy()
                }
                ai_config["ai_configs"][player] = player_config
                config_changed = True
            ai_type = player_config.get("ai_type", "none")
            if ai_type != "none" and ai_type in valid_ai_types:
                ai_code = valid_ai_types[ai_type].get("code", "")
                player_config["ai_code"] = ai_code
                ai_specific_config = self.load_ai_specific_config(ai_code)
                params = ai_specific_config.get(player, DEFAULT_AI_PARAMS.copy())
                for param_type, param_values in DEFAULT_AI_PARAMS.items():
                    if param_type not in params:
                        params[param_type] = param_values.copy() if isinstance(param_values, dict) else param_values
                        config_changed = True
                    elif isinstance(param_values, dict):
                        for key, value in param_values.items():
                            if key not in params[param_type]:
                                params[param_type][key] = value
                                config_changed = True
                player_config["params"] = params
            else:
                player_config["ai_code"] = None
                player_config["params"] = DEFAULT_AI_PARAMS.copy()
            ai_config["ai_configs"][player] = player_config

        if config_changed:
            logger.debug("Changes detected, saving AI config")
            self.save_ai_config(ai_config)

        self._ai_config_cache = ai_config
        logger.debug("AI config cached successfully")
        return self._ai_config_cache

    def save_ai_config(self, ai_config: Dict):
        ai_config_path = self.get_ai_config_path()
        try:
            def make_serializable(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                return obj

            serializable_config = make_serializable(ai_config)
            ai_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ai_config_path, "w", encoding="utf-8") as f:
                json.dump(serializable_config, f, ensure_ascii=False, indent=4)
            logger.debug(f"AI config saved to {ai_config_path}")
            self._ai_config_cache = ai_config
        except Exception as save_ai_config_error:
            logger.error(f"Error saving AI config to {ai_config_path}: {save_ai_config_error}")

    def load_ai_specific_config(self, ai_code: str) -> Dict:
        if ai_code in self._ai_specific_config_cache:
            logger.debug(f"Returning cached AI specific config for {ai_code}")
            return self._ai_specific_config_cache[ai_code]

        ai_specific_config_path = self.get_ai_specific_config_path(ai_code)
        try:
            if ai_specific_config_path.exists():
                with open(ai_specific_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    if not isinstance(config, dict):
                        raise ValueError(f"AI specific config file {ai_specific_config_path} must contain a dictionary")
                    for player in ["player_1", "player_2"]:
                        if player not in config:
                            config[player] = DEFAULT_AI_PARAMS.copy()
                        else:
                            for param_type, param_values in DEFAULT_AI_PARAMS.items():
                                if param_type not in config[player]:
                                    config[player][param_type] = param_values.copy()
                                else:
                                    for key, value in param_values.items():
                                        if key not in config[player][param_type]:
                                            config[player][param_type][key] = value
                    logger.debug(f"Loaded AI specific config from {ai_specific_config_path}")
            else:
                logger.debug(f"AI specific config file not found at {ai_specific_config_path}, creating with default config")
                config = {
                    "player_1": DEFAULT_AI_PARAMS.copy(),
                    "player_2": DEFAULT_AI_PARAMS.copy()
                }
                self.save_ai_specific_config(ai_code, config)
            self._ai_specific_config_cache[ai_code] = config
            logger.debug(f"AI specific config cached for {ai_code}")
        except (json.JSONDecodeError, ValueError, Exception) as ai_specific_error:
            logger.error(f"Error loading AI specific config from {ai_specific_config_path}: {ai_specific_error}, using default config")
            config = {
                "player_1": DEFAULT_AI_PARAMS.copy(),
                "player_2": DEFAULT_AI_PARAMS.copy()
            }
            self._ai_specific_config_cache[ai_code] = config
            self.save_ai_specific_config(ai_code, config)
        return self._ai_specific_config_cache[ai_code]

    def save_ai_specific_config(self, ai_code: str, config: Dict):
        ai_specific_config_path = self.get_ai_specific_config_path(ai_code)
        try:
            ai_specific_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ai_specific_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger.debug(f"AI specific config saved to {ai_specific_config_path}")
            self._ai_specific_config_cache[ai_code] = config
        except Exception as save_ai_specific_error:
            logger.error(f"Error saving AI specific config to {ai_specific_config_path}: {save_ai_specific_error}")

    def get_ai_module(self, ai_type: str):
        if ai_type in self._ai_module_cache:
            return self._ai_module_cache[ai_type].load()
        raise ValueError(f"AI type {ai_type} not found in module cache")

    def load_stats(self) -> Dict:
        stats_path = self.get_stats_path()
        default_stats = {
            "player_1_wins": 0,
            "player_2_wins": 0,
            "ai_stats": {}
        }
        try:
            if stats_path.exists():
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                    if not isinstance(stats, dict):
                        raise ValueError("Stats file must contain a dictionary")
                    default_stats.update(stats)
            else:
                logger.debug(f"Stats file not found at {stats_path}, creating with default stats")
                self.save_stats(default_stats)
        except (json.JSONDecodeError, ValueError, Exception) as stats_error:
            logger.error(f"Error loading stats from {stats_path}: {stats_error}, using default stats")
            self.save_stats(default_stats)
        return default_stats

    def save_stats(self, stats: Dict):
        stats_path = self.get_stats_path()
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)
            logger.debug(f"Stats saved to {stats_path}")
        except Exception as save_stats_error:
            logger.error(f"Error saving stats to {stats_path}: {save_stats_error}")

    def update_ability_level(self, player: str, selected_level: str, language: str):
        try:
            ai_config = self.load_ai_config()
            ability_levels = self.get_ability_levels_reverse(language)
            level_value = int(ability_levels.get(selected_level, "5"))
            ai_config["ai_configs"][player]["ability_level"] = level_value
            self.save_ai_config(ai_config)
            logger.debug(f"Updated ability level for {player} to {level_value} ({selected_level})")
        except Exception as ability_error:
            logger.error(f"Error updating ability level for {player}: {ability_error}")

    def get_ability_level_label(self, player: str, language: str) -> str:
        try:
            ai_config = self.load_ai_config()
            ability_level = ai_config["ai_configs"].get(player, {}).get("ability_level", 5)
            ability_key = ai_config["ability_levels"].get(str(ability_level), "medium")
            return LANGUAGES[language].get(ability_key, "Medium")
        except Exception as label_error:
            logger.error(f"Error getting ability level label for {player}: {label_error}")
            return LANGUAGES[language].get("medium", "Medium")

    def get_ability_levels_reverse(self, language: str) -> Dict[str, str]:
        try:
            ai_config = self.load_ai_config()
            return {
                LANGUAGES[language][key]: str(value) for value, key in ai_config["ability_levels"].items()
            }
        except Exception as reverse_error:
            logger.error(f"Error getting reverse ability levels for language {language}: {reverse_error}")
            return {
                LANGUAGES[language][key]: str(value) for value, key in DEFAULT_ABILITY_LEVELS.items()
            }

# Wrapper functions for backward compatibility
_config_manager = ConfigManager()

def get_project_root() -> Path:
    return _config_manager.get_project_root()

def get_stats_path() -> Path:
    return _config_manager.get_stats_path()

def get_config_path() -> Path:
    return _config_manager.get_config_path()

def get_ai_config_path() -> Path:
    return _config_manager.get_ai_config_path()

def get_ai_specific_config_path(ai_code: str) -> Path:
    return _config_manager.get_ai_specific_config_path(ai_code)

def load_config() -> Dict:
    return _config_manager.load_config()

def save_config(config: Dict):
    _config_manager.save_config(config)

def load_ai_config() -> Dict:
    return _config_manager.load_ai_config()

def save_ai_config(ai_config: Dict):
    _config_manager.save_ai_config(ai_config)

def load_ai_specific_config(ai_code: str) -> Dict:
    return _config_manager.load_ai_specific_config(ai_code)

def save_ai_specific_config(ai_code: str, config: Dict):
    _config_manager.save_ai_specific_config(ai_code, config)

def load_stats() -> Dict:
    return _config_manager.load_stats()

def save_stats(stats: Dict):
    _config_manager.save_stats(stats)

def log_to_json(message: str, level: str = "DEBUG", extra_data: Optional[Dict] = None):
    ConfigManager.log_to_json(message, level, extra_data)


