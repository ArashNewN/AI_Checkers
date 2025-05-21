# config.py
import json
import sys
from pathlib import Path
import importlib
import logging
from functools import partial

logger = logging.getLogger(__name__)

# کلاس LazyLoader برای بارگذاری تنبل ماژول‌ها
class LazyLoader:
    def __init__(self, module_name, class_name):
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
            except Exception as e:
                logger.error(f"Error lazy-loading module {self.module_name}: {e}")
                raise
        return self._class

# کلاس ConfigManager برای مدیریت مرکزی تنظیمات
class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._config_cache = None
        self._ai_config_cache = None
        self._ai_specific_config_cache = {}
        self._ai_module_cache = {}
        self.project_root = None

    def get_project_root(self) -> Path:
        if self.project_root is None:
            current_path = Path(__file__).resolve().parent
            max_depth = 10
            depth = 0
            while depth < max_depth:
                configs_dir = current_path / "configs"
                a_dir = current_path / "a"
                # ایجاد خودکار پوشه configs اگر وجود نداشته باشد
                configs_dir.mkdir(parents=True, exist_ok=True)
                # بررسی وجود پوشه‌های موردنیاز
                if configs_dir.exists() and a_dir.exists():
                    logger.debug(f"Project root found at: {current_path}")
                    self.project_root = current_path
                    return self.project_root
                current_path = current_path.parent
                depth += 1
            logger.error("Could not find project root with config and a directories")
            raise FileNotFoundError("Failed to locate project root directory")
        return self.project_root

    def get_stats_path(self):
        """Returns the path to stats.json."""
        return self.get_project_root() / "configs" / "stats.json"  # تغییر مسیر به configs

    def get_config_path(self):
        """Returns the path to config.json."""
        return self.get_project_root() / "configs" / "config.json"  # تغییر مسیر به configs

    def get_ai_config_path(self):
        """Returns the path to ai_config.json."""
        return self.get_project_root() / "configs" / "ai_config.json"

    def get_ai_specific_config_path(self, ai_code):
        """Returns the path to AI-specific config (e.g., al_config.json)."""
        return self.get_project_root() / "configs" / "ai" / f"{ai_code}_config.json"

    def load_config(self):
        """Loads non-AI settings from config.json or creates it with defaults."""
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
            "player_1_piece_image": "",
            "player_1_king_image": "",
            "player_2_piece_image": "",
            "player_2_king_image": "",
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
            "assets_dir": str(self.get_project_root() / "assets"),
            "max_no_capture_moves": 40,
            "max_uniform_moves": 5,
            "max_total_moves": 40,
            "logging_level": "ERROR"
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
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.error(f"Error loading config from {config_path}: {e}, using default config")
            config = default_config
            self._config_cache = config
            self.save_config(config)
        return self._config_cache

    def save_config(self, config):
        """Saves non-AI settings to config.json."""
        config_path = self.get_config_path()
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.debug(f"Config saved to {config_path}")
            self._config_cache = config  # Update cache after saving
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")

    def load_ai_config(self):
        """Loads AI settings from ai_config.json and separate config files."""
        if self._ai_config_cache is not None:
            logger.debug("Returning cached AI config")
            return self._ai_config_cache

        default_ai_config = {
            "ai_types": {},
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
                    ai_config = json.load(f)
                    if not isinstance(ai_config, dict):
                        raise ValueError("AI config file must contain a dictionary")
                    for key, value in default_ai_config.items():
                        if key not in ai_config:
                            ai_config[key] = value
                            config_changed = True
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if sub_key not in ai_config[key]:
                                    ai_config[key][sub_key] = sub_value
                                    config_changed = True
                    logger.debug(f"Loaded AI config from {ai_config_path}")
            else:
                logger.debug(f"AI config file not found at {ai_config_path}, creating with default config")
                config_changed = True
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.error(f"Error loading AI config from {ai_config_path}: {e}, using default AI config")
            ai_config = default_ai_config
            config_changed = True

        # Validate AI types
        used_codes = set()
        valid_ai_types = {}
        for ai_type, ai_info in ai_config["ai_types"].items():
            code = ai_info.get("code")
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

        # Validate and cache AI modules
        project_dir = self.get_project_root()
        root_dir = project_dir
        if str(root_dir) not in sys.path:
            sys.path.append(str(root_dir))

        valid_ai_types = {}
        for ai_type, ai_info in ai_config["ai_types"].items():
            full_module_name = ai_info.get("module", "")
            if not full_module_name.startswith("a."):
                logger.warning(f"AI type {ai_type} ignored: invalid module name {full_module_name}")
                continue
            module_name = full_module_name.replace("a.", "")
            module_path = project_dir / f"{module_name}.py"
            class_name = ai_info.get("class", "")

            logger.debug(f"Checking module: {module_path} for AI type {ai_type}")
            if module_path.exists():
                # Cache LazyLoader instead of loading module immediately
                self._ai_module_cache[ai_type] = LazyLoader(full_module_name, class_name)
                valid_ai_types[ai_type] = ai_info
                logger.debug(f"AI type {ai_type} prepared for lazy loading: module {full_module_name}, class {class_name}")
                # Create separate AI config file if new
                ai_code = ai_info.get("code")
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

        # Load AI-specific settings
        for player in ["player_1", "player_2"]:
            player_config = ai_config["ai_configs"].get(player, {})
            ai_type = player_config.get("ai_type", "none")
            if ai_type != "none" and ai_type in valid_ai_types:
                ai_code = valid_ai_types[ai_type]["code"]
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

    def save_ai_config(self, ai_config):
        """Saves AI settings to ai_config.json."""
        ai_config_path = self.get_ai_config_path()
        try:
            ai_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ai_config_path, "w", encoding="utf-8") as f:
                json.dump(ai_config, f, ensure_ascii=False, indent=4)
            logger.debug(f"AI config saved to {ai_config_path}")
            self._ai_config_cache = ai_config  # Update cache after saving
        except Exception as e:
            logger.error(f"Error saving AI config to {ai_config_path}: {e}")

    def load_ai_specific_config(self, ai_code):
        """Loads AI-specific config from its config file (e.g., al_config.json)."""
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
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.error(f"Error loading AI specific config from {ai_specific_config_path}: {e}, using default config")
            config = {
                "player_1": DEFAULT_AI_PARAMS.copy(),
                "player_2": DEFAULT_AI_PARAMS.copy()
            }
            self._ai_specific_config_cache[ai_code] = config
            self.save_ai_specific_config(ai_code, config)
        return self._ai_specific_config_cache[ai_code]

    def save_ai_specific_config(self, ai_code, config):
        """Saves AI-specific settings to its config file (e.g., al_config.json)."""
        ai_specific_config_path = self.get_ai_specific_config_path(ai_code)
        try:
            ai_specific_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ai_specific_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logger.debug(f"AI specific config saved to {ai_specific_config_path}")
            self._ai_specific_config_cache[ai_code] = config  # Update cache after saving
        except Exception as e:
            logger.error(f"Error saving AI specific config to {ai_specific_config_path}: {e}")

    def get_ai_module(self, ai_type):
        """Returns the lazily loaded AI module class."""
        if ai_type in self._ai_module_cache:
            return self._ai_module_cache[ai_type].load()
        raise ValueError(f"AI type {ai_type} not found in module cache")

    def load_stats(self):
        """Loads game statistics."""
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
        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.error(f"Error loading stats from {stats_path}: {e}, using default stats")
            self.save_stats(default_stats)
        return default_stats

    def save_stats(self, stats):
        """Saves game statistics."""
        stats_path = self.get_stats_path()
        try:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)
            logger.debug(f"Stats saved to {stats_path}")
        except Exception as e:
            logger.error(f"Error saving stats to {stats_path}: {e}")

# ثابت‌های تنظیمات پیش‌فرض هوش مصنوعی
DEFAULT_AI_PARAMS = {
    "ability_level": 5,
    "training_params": {
        "memory_size": 10000,
        "batch_size": 128,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.999,
        "update_target_every": 100,
        "reward_threshold": 0.5,
        "weight_decay": 0.0,
        "gradient_clip": 1.0,
        "target_update_alpha": 0.01
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

# توابع wrapper برای سازگاری با کد موجود
_config_manager = ConfigManager()

def get_project_root():
    return _config_manager.get_project_root()

def get_stats_path():
    return _config_manager.get_stats_path()

def get_config_path():
    return _config_manager.get_config_path()

def get_ai_config_path():
    return _config_manager.get_ai_config_path()

def get_ai_specific_config_path(ai_code):
    return _config_manager.get_ai_specific_config_path(ai_code)

def load_config():
    return _config_manager.load_config()

def save_config(config):
    _config_manager.save_config(config)

def load_ai_config():
    return _config_manager.load_ai_config()

def save_ai_config(ai_config):
    _config_manager.save_ai_config(ai_config)

def load_ai_specific_config(ai_code):
    return _config_manager.load_ai_specific_config(ai_code)

def save_ai_specific_config(ai_code, config):
    _config_manager.save_ai_specific_config(ai_code, config)

def load_stats():
    return _config_manager.load_stats()

def save_stats(stats):
    _config_manager.save_stats(stats)