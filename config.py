import json
import os
import sys
from pathlib import Path
import importlib

LANGUAGES = {
    "en": {
        "piece_images": "Piece Images",
        "player_1_piece": "Player 1 Piece",
        "player_1_king": "Player 1 King",
        "player_2_piece": "Player 2 Piece",
        "player_2_king": "Player 2 King",
        "al_progress": "AL Progress",
        "az_progress": "AZ Progress",
        "wins": "Wins",
        "losses": "Losses",
        "add_new_ai": "add_new_ai",
        "draws": "Draws",
        "time": "TIME:",
        "remove_ai": "remove_ai:",
        "avg_move_time": "Avg Move Time (ms)",
        "settings": "Settings",
        "help": "Help",
        "player_1_color": "player_1_color",
        "player_2_color": "player_2_color",
        "about_me": "About me",
        "start_game": "Start Game",
        "new_game": "New Game",
        "reset_scores": "Reset Scores",
        "game_result": "Game Result",
        "player_wins": "Player 1 Wins",
        "ai_wins": "Player 2 Wins",
        "reset_scores_warning": "Are you sure you want to reset scores?",
        "pieces": "Pieces",
        "settings_reset": "Settings have been reset to default values.",
        "footer": "",
        "confirm": "Confirm",
        "new_game_warning": "Start a new game?",
        "warning": "Warning",
        "game_settings_tab": "Game",
        "default_reward_weights": "default_reward_weights",
        "design_tab": "Design",
        "ai_tab": "AI",
        "player_tab": "Players",
        "advanced_ai_settings": "Advanced AI Settings",  # اضافه شده برای رفع خطا
        "play_with": "Play With",
        "human_vs_human": "Human vs Human",
        "human_vs_ai": "Human vs AI",
        "ai_vs_ai": "AI vs AI",
        "only_once": "Only Once",
        "repeat_game": "Repeat Game",
        "starting_player": "Starting Player",
        "player": "Player 1",
        "ai": "Player 2",
        "game_timer": "Game Timer",
        "no_timer": "No Timer",
        "with_timer": "With Timer",
        "game_duration": "Duration (min):",
        "language": "Language",
        "color_settings": "Colors",
        "player_piece_color": "Player 1:",
        "ai_piece_color": "Player 2:",
        "board_color_1": "Board Light:",
        "board_color_2": "Board Dark:",
        "piece_style": "Piece Style",
        "sound_settings": "Sound",
        "sound_on": "On",
        "sound_off": "Off",
        "ai_pause_time": "AI Pause Time",
        "ms": "ms",
        "ai_ability": "AI Ability",
        "player_1_name": "Player 1 Name",
        "player_2_name": "Player 2 Name",
        "al1_name": "AL 1 Name",
        "al2_name": "AL 2 Name",
        "upload_image": "Upload Image",
        "save_changes": "Save",
        "close": "Close",
        "reset_settings": "Reset",
        "coming_soon": "Coming Soon",
        "player_1_ai_type": "Player 1 AI Type",
        "player_2_ai_type": "Player 2 AI Type",
        "error": "Error",
        "ai_pause_error": "AI pause time must be between 0 and 5000 ms",
        "invalid_number_hands": "Number of hands must be between 1 and 1000",
        "invalid_number_error": "Please enter valid numbers",
        "apply_after_game": "Settings will apply after the game ends",
        "reset_settings_warning": "Are you sure you want to reset all settings?",
        "advanced_settings_tab": "Advanced Settings",
        "advanced_settings_warning": "These settings may affect AI performance. If you lack expertise, we recommend using default values.",
        "ai_ability_level": "AI Ability Level",
        "very_weak": "Very Weak",
        "weak": "Weak",
        "ai_type": "ai_type",
        "ai_settings_tab": "ai_settings_tab",
        "add_ai": "add_ai",
        "new_ai_type": "new_ai_type",
        "strong": "Strong",
        "very_strong": "Very Strong",
        "select_module_class": "select_module_class",
        "advanced_config": "Advanced Config",
        "al1_training_params": "AL 1 Training Parameters",
        "al2_training_params": "AL 2 Training Parameters",
        "az1_training_params": "AlphaZero 1 Training Parameters",
        "az2_training_params": "AlphaZero 2 Training Parameters",
        "mcts_params": "MCTS Parameters",
        "medium": "Medium",
        "default_ai_params": "default_ai_params",
        "network_params": "Network Parameters",
        "advanced_nn_params": "Advanced NN Parameters",
        "reset_tab": "Reset This Tab",
        "save_all": "Save All",
        "advanced_settings_title": "Advanced AI Configuration",
        "open_advanced_config": "Open Advanced Configuration",
        "training_params": "Training Parameters",
        "reward_weights": "Reward Weights",
        "description": "description",
        "search_modules": "search_modules",
        "remove_selected": "remove_selected",
        "player_1": "Player 1",
        "new_description": "new_description",
        "player_2": "Player 2",
        "settings_saved": "Settings saved successfully.",
        "ai_added": "AI added successfully.",
        "ai_removed": "AI removed successfully.",
        "fill_all_fields": "Please fill all required fields.",
        "ai_type_exists": "This AI type already exists.",
        "add": "add",
        "ability_levels": "ability_levels",
        "status": "status",
        "player_2_ability": "player_2_ability",
        "player_1_ability": "player_1_ability",
        "remove": "remove",
        "info": "info",
        "ability": "ability",
        "ai_pause_time_ms": "ai_pause_time_ms",
        "player_1_ai": "player_1_ai",
        "player_2_ai": "player_2_ai",
        "select_ai_to_remove": "Please select an AI to remove.",
        "ai_players": "ai_players",
        "no_ai_selected": "No AI selected",
        "sound_enabled": "sound_enabled",
        "invalid_input": "Invalid input. Please enter a valid value.",
    },
    "fa": {
        "piece_images": "تصاویر مهره‌ها",
        "sound_enabled": "sound_enabled",
        "player_1_piece": "مهره بازیکن ۱",
        "player_1_king": "شاه بازیکن ۱",
        "info": "info",
        "no_ai_selected": "No AI selected",
        "player_2_piece": "مهره بازیکن ۲",
        "player_2_king": "شاه بازیکن ۲",
        "add_ai": "add_ai",
        "ability": "ability",
        "status": "status",
        "ai_players": "ai_players",
        "player_1_ai": "player_1_ai",
        "player_2_ai": "player_2_ai",
        "ai_pause_time_ms": "ai_pause_time_ms",
        "player_1_color": "player_1_color",
        "player_2_color": "player_2_color",
        "add": "add",
        "player_2_ability": "player_2_ability",
        "player_1_ability": "player_1_ability",
        "remove": "remove",
        "al_progress": "پیشرفت هوش مصنوعی",
        "remove_selected": "remove_selected",
        "az_progress": "پیشرفت هوش مصنوعی",
        "ai_type": "ai_type",
        "search_modules": "search_modules",
        "description": "description",
        "add_new_ai": "add_new_ai",
        "settings_reset": "تنظیمات به مقادیر پیش‌فرض بازنشانی شدند.",
        "ai_added": "هوش مصنوعی با موفقیت اضافه شد.",
        "ai_settings_tab": "ai_settings_tab",
        "remove_ai": "remove_ai:",
        "ai_removed": "هوش مصنوعی با موفقیت حذف شد.",
        "fill_all_fields": "لطفاً تمام فیلدهای مورد نیاز را پر کنید.",
        "ai_type_exists": "این نوع هوش مصنوعی قبلاً وجود دارد.",
        "select_ai_to_remove": "لطفاً یک هوش مصنوعی برای حذف انتخاب کنید.",
        "wins": "بردها",
        "losses": "باخت‌ها",
        "draws": "تساوی‌ها",
        "time": "زمان:",
        "avg_move_time": "میانگین زمان حرکت (میلی‌ثانیه)",
        "settings": "تنظیمات",
        "help": "راهنما",
        "about_me": "درباره من",
        "start_game": "شروع بازی",
        "new_game": "بازی جدید",
        "reset_scores": "بازنشانی امتیازات",
        "game_result": "نتیجه بازی",
        "ability_levels": "ability_levels",
        "player_wins": "بازیکن ۱ برنده شد",
        "ai_wins": "بازیکن ۲ برنده شد",
        "reset_scores_warning": "آیا مطمئن هستید که می‌خواهید امتیازات را بازنشانی کنید؟",
        "pieces": "مهره‌ها",
        "footer": "",
        "new_description": "new_description",
        "confirm": "تأیید",
        "new_game_warning": "آیا بازی جدیدی شروع شود؟",
        "warning": "هشدار",
        "game_settings_tab": "بازی",
        "design_tab": "طراحی",
        "ai_tab": "هوش مصنوعی",
        "advanced_settings_title": "پیکربندی پیشرفته هوش مصنوعی",
        "open_advanced_config": "باز کردن پیکربندی پیشرفته",
        "training_params": "پارامترهای آموزش",
        "reward_weights": "وزن‌های پاداش",
        "player_1": "بازیکن ۱",
        "player_2": "بازیکن ۲",
        "player_tab": "بازیکنان",
        "advanced_ai_settings": "تنظیمات پیشرفته هوش مصنوعی",  # اضافه شده برای رفع خطا
        "play_with": "بازی با",
        "human_vs_human": "انسان در برابر انسان",
        "human_vs_ai": "انسان در برابر هوش مصنوعی",
        "ai_vs_ai": "هوش مصنوعی در برابر هوش مصنوعی",
        "new_ai_type": "new_ai_type",
        "only_once": "فقط یک بار",
        "repeat_game": "تکرار بازی",
        "starting_player": "بازیکن شروع‌کننده",
        "player": "بازیکن ۱",
        "ai": "بازیکن ۲",
        "game_timer": "تایمر بازی",
        "no_timer": "بدون تایمر",
        "with_timer": "با تایمر",
        "game_duration": "مدت زمان (دقیقه):",
        "language": "زبان",
        "color_settings": "تنظیمات رنگ",
        "player_piece_color": "بازیکن ۱:",
        "ai_piece_color": "بازیکن ۲:",
        "board_color_1": "روشن صفحه:",
        "board_color_2": "تیره صفحه:",
        "piece_style": "سبک مهره",
        "sound_settings": "صدا",
        "sound_on": "روشن",
        "sound_off": "خاموش",
        "ai_pause_time": "زمان مکث هوش مصنوعی",
        "ms": "میلی‌ثانیه",
        "ai_ability": "توانایی هوش مصنوعی",
        "player_1_name": "نام بازیکن ۱",
        "player_2_name": "نام بازیکن ۲",
        "al1_name": "نام هوش مصنوعی ۱",
        "al2_name": "نام هوش مصنوعی ۲",
        "upload_image": "بارگذاری تصویر",
        "save_changes": "ذخیره",
        "close": "بستن",
        "reset_settings": "بازنشانی",
        "default_reward_weights": "default_reward_weights",
        "coming_soon": "به زودی",
        "player_1_ai_type": "نوع هوش مصنوعی بازیکن ۱",
        "player_2_ai_type": "نوع هوش مصنوعی بازیکن ۲",
        "error": "خطا",
        "ai_pause_error": "زمان مکث هوش مصنوعی باید بین ۰ تا ۵۰۰۰ میلی‌ثانیه باشد",
        "invalid_number_hands": "تعداد دست‌ها باید بین ۱ تا ۱۰۰۰ باشد",
        "invalid_number_error": "لطفاً اعداد معتبر وارد کنید",
        "apply_after_game": "تنظیمات پس از پایان بازی اعمال خواهند شد",
        "reset_settings_warning": "آیا مطمئن هستید که می‌خواهید تمام تنظیمات را بازنشانی کنید؟",
        "advanced_settings_tab": "تنظیمات پیشرفته",
        "advanced_settings_warning": "این تنظیمات ممکن است روی کارایی هوش‌های مصنوعی تأثیر بگذارد. در صورتی که در این زمینه آگاهی ندارید، پیشنهاد می‌کنیم از گزینه‌های پیش‌فرض استفاده کنید.",
        "ai_ability_level": "سطح توانایی هوش مصنوعی",
        "very_weak": "خیلی ضعیف",
        "weak": "ضعیف",
        "medium": "متوسط",
        "default_ai_params": "default_ai_params",
        "strong": "قوی",
        "settings_saved": "تنظیمات با موفقیت ذخیره شدند.",
        "very_strong": "خیلی قوی",
        "advanced_config": "پیکربندی پیشرفته",
        "al1_training_params": "پارامترهای آموزش Advanced AI ۱",
        "al2_training_params": "پارامترهای آموزش Advanced AI ۲",
        "az1_training_params": "پارامترهای آموزش AlphaZero ۱",
        "az2_training_params": "پارامترهای آموزش AlphaZero ۲",
        "mcts_params": "پارامترهای MCTS",
        "select_module_class": "select_module_class",
        "network_params": "پارامترهای شبکه",
        "advanced_nn_params": "پارامترهای شبکه پیشرفته",
        "reset_tab": "بازنشانی این تب",
        "save_all": "ذخیره همه",
        "invalid_input": "ورودی نامعتبر. لطفاً مقدار معتبر وارد کنید."
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
    return Path(__file__).parent.parent / "ai_config.json"

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
        "window_height": 727,
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
        "max_no_capture_moves": 40,
        "ai_configs": {
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
        },
        "default_ai_params": {
            "training_params": {
                "memory_size": 10000,
                "batch_size": 128,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.999
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
            }
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
    config_path = get_config_path()
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                # به‌روزرسانی تنظیمات موجود با پیش‌فرض‌ها
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if sub_key not in config[key]:
                                config[key][sub_key] = sub_value
                print(f"Loaded config from {config_path}")
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
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {config_path}")

def load_ai_config():
    """بارگذاری تنظیمات AI از فایل ai_config.json یا ایجاد آن با تنظیمات پیش‌فرض"""
    default_ai_config = {
        "ai_types": {},  # بدون AI‌های پیش‌فرض
        "ai_configs": {
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
        },
        "default_ai_params": {
            "training_params": {
                "memory_size": 10000,
                "batch_size": 128,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.999
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
            }
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

    ai_config_path = get_ai_config_path()
    try:
        if ai_config_path.exists():
            with open(ai_config_path, "r", encoding="utf-8") as f:
                ai_config = json.load(f)
                # به‌روزرسانی تنظیمات موجود با پیش‌فرض‌ها
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

    # تنظیم مسیر دایرکتوری پروژه و بررسی ماژول‌های AI
    project_dir = Path(__file__).parent
    root_dir = project_dir.parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

    valid_ai_types = {}
    for ai_type, ai_info in ai_config.get("ai_types", {}).items():
        module_name = ai_info.get("module", "")
        full_module_name = f"a.{module_name}"
        module_path = project_dir / f"{module_name}.py"

        print(f"Checking module: {module_path}")
        if module_path.exists():
            try:
                module = importlib.import_module(full_module_name)
                if hasattr(module, ai_info.get("class", "")):
                    valid_ai_types[ai_type] = ai_info
                else:
                    print(f"AI type {ai_type} ignored: class '{ai_info.get('class', '')}' not found in module {module_name}")
            except Exception as e:
                print(f"Error importing module {module_name}: {str(e)}")
        else:
            print(f"AI type {ai_type} ignored: module {module_name}.py not found")

    ai_config["ai_types"] = valid_ai_types
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

def load_stats():
    """بارگذاری آمار بازی"""
    stats_path = get_stats_path()
    default_stats = {
        "player_1_wins": 0,
        "player_2_wins": 0,
        "ai_stats": {
            "al1": {"wins": 0, "losses": 0, "draws": 0, "move_times": []},
            "al2": {"wins": 0, "losses": 0, "draws": 0, "move_times": []}
        }
    }
    try:
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                default_stats.update(stats)
                print(f"Loaded stats from {stats_path}")
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