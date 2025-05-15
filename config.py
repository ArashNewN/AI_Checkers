import json
import os
import sys
from pathlib import Path
import importlib


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

def get_stats_path():
    """بازگرداندن مسیر فایل stats.json"""
    return Path(__file__).parent.parent / "stats.json"

def get_config_path():
    """بازگرداندن مسیر فایل config.json"""
    return Path(__file__).parent.parent / "config.json"

def get_ai_config_path():
    """بازگرداندن مسیر فایل ai_config.json"""
    return Path(__file__).parent.parent / "configs" / "ai_config.json"

def get_ai_specific_config_path(ai_code):
    """بازگرداندن مسیر فایل کانفیگ خاص AI (مثل al_config.json)"""
    return Path(__file__).parent.parent / "configs" / "ai" / f"{ai_code}_config.json"

def load_config():
    """بارگذاری تنظیمات غیر AI از فایل config.json یا ایجاد آن با تنظیمات پیش‌فرض"""
    default_config = {
        # ثابت‌های رابط کاربری
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
        # تنظیمات جدید برای Hint
        "hint_enabled_p1_default": False,
        "hint_enabled_p2_default": False,
        "hint_circle_color": [255, 165, 0],  # نارنجی
        "hint_circle_radius": 10,
        "hint_blink_interval": 500,  # میلی‌ثانیه
        "hint_button_width": 120,
        "hint_button_height": 40,
        "hint_button_spacing": 10,
        "hint_button_y_offset": 10,
        # تنظیمات جدید برای Undo/Redo
        "undo_button_width": 120,
        "undo_button_height": 40,
        "redo_button_width": 120,
        "redo_button_height": 40,
        "undo_redo_button_spacing": 10,
        "undo_redo_y_offset": 10,
        # رنگ‌ها
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "red": [255, 0, 0],
        "gray": [128, 128, 128],
        "blue": [0, 0, 255],
        "sky_blue": [135, 206, 235],
        "light_gray": [211, 211, 211],
        "light_green": [144, 238, 144],
        # تنظیمات بازی
        "piece_style": "circle",
        "sound_enabled": False,
        "ai_pause_time": 20,
        "game_mode": "human_vs_human",
        "ai_vs_ai_mode": "only_once",
        "repeat_hands": 10,
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
        "use_timer": True,
        "game_time": 5,
        "language": "en",
        "player_1_color": "#ff0000",
        "player_2_color": "#0000ff",
        "board_color_1": "#ffffff",
        "board_color_2": "#8b4513",
        "max_no_capture_moves": 40
    }
    config_path = get_config_path()
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in config[key]:
                                config[key][sub_key] = sub_value
        else:
            print(f"Config file not found at {config_path}, creating with default config")
            config = default_config
            save_config(config)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading config from {config_path}: {e}, using default config")
        config = default_config
        save_config(config)
    return config

def save_config(config):
    """ذخیره تنظیمات غیر AI در فایل config.json"""
    config_path = get_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")

def load_ai_config():
    """بارگذاری تنظیمات AI از ai_config.json و فایل‌های کانفیگ مجزا"""
    default_ai_config = {
        "ai_types": {},  # خالی، بدون AI پیش‌فرض
        "ai_configs": {
            "player_1": {
                "ai_type": "none",
                "ai_code": None,
                "ability_level": 5,
                "params": {}
            },
            "player_2": {
                "ai_type": "none",
                "ai_code": None,
                "ability_level": 5,
                "params": {}
            }
        },
        "available_ais": []  # خالی، بدون AI پیش‌فرض
    }

    ai_config_path = get_ai_config_path()
    try:
        if ai_config_path.exists():
            with open(ai_config_path, "r", encoding="utf-8") as f:
                ai_config = json.load(f)
                for key, value in default_ai_config.items():
                    if key not in ai_config:
                        ai_config[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in ai_config[key]:
                                ai_config[key][sub_key] = sub_value
                print(f"Loaded AI config from {ai_config_path}")
        else:
            print(f"AI config file not found at {ai_config_path}, creating with default config")
            ai_config = default_ai_config
            save_ai_config(ai_config)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading AI config from {ai_config_path}: {e}, using default AI config")
        ai_config = default_ai_config
        save_ai_config(ai_config)

    # اعتبارسنجی کدهای دوحرفی
    used_codes = set()
    for ai in ai_config["available_ais"]:
        code = ai.get("code")
        if not code or len(code) != 2:
            print(f"Invalid or missing code for AI {ai['type']}, skipping")
            continue
        if code in used_codes:
            print(f"Duplicate code {code} for AI {ai['type']}, skipping")
            continue
        used_codes.add(code)

    # اضافه کردن AIهای جدید به ai_types
    for ai in ai_config["available_ais"]:
        ai_type = ai.get("type")
        if ai_type and ai_type not in ai_config["ai_types"]:
            ai_config["ai_types"][ai_type] = {
                "module": ai.get("module"),
                "class": ai.get("class"),
                "code": ai.get("code")
            }

    # اعتبارسنجی ماژول‌های AI
    project_dir = Path(__file__).parent
    root_dir = project_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

    valid_ai_types = {}
    for ai_type, ai_info in ai_config["ai_types"].items():
        full_module_name = ai_info.get("module", "")
        if not full_module_name.startswith("a."):
            print(f"AI type {ai_type} ignored: invalid module name {full_module_name}")
            continue
        module_name = full_module_name.replace("a.", "")
        module_path = project_dir / f"{module_name}.py"

        print(f"Checking module: {module_path} for AI type {ai_type}")
        if module_path.exists():
            try:
                module = importlib.import_module(full_module_name)
                class_name = ai_info.get("class", "")
                if hasattr(module, class_name):
                    valid_ai_types[ai_type] = ai_info
                    print(f"AI type {ai_type} validated: module {full_module_name}, class {class_name}")
                    # ایجاد فایل کانفیگ مجزا برای AI
                    ai_code = ai_info.get("code")
                    if ai_code:
                        ai_specific_config_path = get_ai_specific_config_path(ai_code)
                        if not ai_specific_config_path.exists():
                            save_ai_specific_config(ai_code, {
                                "player_1": DEFAULT_AI_PARAMS.copy(),
                                "player_2": DEFAULT_AI_PARAMS.copy()
                            })
                else:
                    print(f"AI type {ai_type} ignored: class '{class_name}' not found in module {full_module_name}")
            except Exception as e:
                print(f"AI type {ai_type} ignored: error importing module {full_module_name}: {str(e)}")
        else:
            print(f"AI type {ai_type} ignored: module file {module_path} not found")

    ai_config["ai_types"] = valid_ai_types

    # لود تنظیمات خاص AIها از فایل‌های کانفیگ مجزا
    for player in ["player_1", "player_2"]:
        ai_type = ai_config["ai_configs"][player]["ai_type"]
        if ai_type != "none" and ai_type in valid_ai_types:
            ai_code = valid_ai_types[ai_type]["code"]
            ai_config["ai_configs"][player]["ai_code"] = ai_code
            ai_specific_config = load_ai_specific_config(ai_code)
            ai_config["ai_configs"][player]["params"] = ai_specific_config.get(player, DEFAULT_AI_PARAMS.copy())
        else:
            ai_config["ai_configs"][player]["ai_code"] = None
            ai_config["ai_configs"][player]["params"] = {}

    save_ai_config(ai_config)
    return ai_config

def save_ai_config(ai_config):
    """ذخیره تنظیمات AI در فایل ai_config.json"""
    ai_config_path = get_ai_config_path()
    try:
        ai_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ai_config_path, "w", encoding="utf-8") as f:
            json.dump(ai_config, f, ensure_ascii=False, indent=4)
        print(f"AI config saved to {ai_config_path}")
    except Exception as e:
        print(f"Error saving AI config to {ai_config_path}: {e}")

def load_ai_specific_config(ai_code):
    ai_specific_config_path = get_ai_specific_config_path(ai_code)
    try:
        if ai_specific_config_path.exists():
            with open(ai_specific_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
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
                #print(f"Loaded AI specific config from {ai_specific_config_path}: {config}")
        else:
            print(f"AI specific config file not found at {ai_specific_config_path}, creating with default config")
            config = {
                "player_1": DEFAULT_AI_PARAMS.copy(),
                "player_2": DEFAULT_AI_PARAMS.copy()
            }
            save_ai_specific_config(ai_code, config)
        return config
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading AI specific config from {ai_specific_config_path}: {e}, using default config")
        config = {
            "player_1": DEFAULT_AI_PARAMS.copy(),
            "player_2": DEFAULT_AI_PARAMS.copy()
        }
        save_ai_specific_config(ai_code, config)
        return config

def save_ai_specific_config(ai_code, config):
    """ذخیره تنظیمات خاص AI در فایل کانفیگ خودش (مثل al_config.json)"""
    ai_specific_config_path = get_ai_specific_config_path(ai_code)
    try:
        ai_specific_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ai_specific_config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"AI specific config saved to {ai_specific_config_path}")
    except Exception as e:
        print(f"Error saving AI specific config to {ai_specific_config_path}: {e}")

def load_stats():
    """بارگذاری آمار بازی"""
    stats_path = get_stats_path()
    default_stats = {
        "player_1_wins": 0,
        "player_2_wins": 0,
        "ai_stats": {}
    }
    try:
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                default_stats.update(stats)
        else:
            print(f"Stats file not found at {stats_path}, creating with default stats")
            save_stats(default_stats)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading stats from {stats_path}: {e}, using default stats")
        save_stats(default_stats)
    return default_stats

def save_stats(stats):
    """ذخیره آمار بازی"""
    stats_path = get_stats_path()
    try:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"Stats saved to {stats_path}")
    except Exception as e:
        print(f"Error saving stats to {stats_path}: {e}")