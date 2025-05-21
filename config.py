import json
from pathlib import Path

# دیکشنری زبان‌ها (ادغام شده از constants.py و نسخه قبلی config.py)
LANGUAGES = {
    "en": {
        "piece_images": "Piece Images",
        "player_1_piece": "Player 1 Piece",
        "player_1_king": "Player 1 King",
        "player_2_piece": "Player 2 Piece",
        "player_2_king": "Player 2 King",
        "ai_progress": "AI Progress",
        "az_progress": "AZ Progress",
        "wins": "Wins",
        "losses": "Losses",
        "draws": "Draws",
        "time": "TIME:",
        "avg_move_time": "Avg Move Time (ms)",
        "settings": "Settings",
        "help": "Help",
        "about_me": "About me",
        "start_game": "Start Game",
        "new_game": "New Game",
        "reset_scores": "Reset Scores",
        "game_result": "Game Result",
        "player_wins": "Player 1 Wins",
        "ai_wins": "Player 2 Wins",
        "reset_scores_warning": "Are you sure you want to reset scores?",
        "pieces": "Pieces",
        "footer": "",
        "confirm": "Confirm",
        "new_game_warning": "Start a new game?",
        "warning": "Warning",
        "game_settings_tab": "Game",
        "design_tab": "Design",
        "ai_tab": "AI",
        "player_tab": "Players",
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
        "ai_1_name": "AI 1 Name",
        "ai_2_name": "AI 2 Name",
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
        # متن‌های جدید برای تب Advanced Settings
        "advanced_settings_tab": "Advanced Settings",
        "advanced_settings_warning": "These settings may affect AI performance. If you lack expertise, we recommend using default values.",
        "ai_ability_level": "AI Ability Level",
        "very_weak": "Very Weak",
        "weak": "Weak",
        "medium": "Medium",
        "strong": "Strong",
        "very_strong": "Very Strong",
        "advanced_config": "Advanced Config",
        "ai_1_training_params": "AI 1 Training Parameters",
        "ai_2_training_params": "AI 2 Training Parameters",
        "az1_training_params": "AlphaZero 1 Training Parameters",
        "az2_training_params": "AlphaZero 2 Training Parameters",
        "mcts_params": "MCTS Parameters",
        "network_params": "Network Parameters",
        "advanced_nn_params": "Advanced NN Parameters",
        "reset_tab": "Reset This Tab",
        "save_all": "Save All",
        "invalid_input": "Invalid input. Please enter a valid value."
    },
    "fa": {
        "piece_images": "تصاویر مهره‌ها",
        "player_1_piece": "مهره بازیکن ۱",
        "player_1_king": "شاه بازیکن ۱",
        "player_2_piece": "مهره بازیکن ۲",
        "player_2_king": "شاه بازیکن ۲",
        "ai_progress": "پیشرفت هوش مصنوعی",
        "az_progress": "پیشرفت هوش مصنوعی",
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
        "player_wins": "بازیکن ۱ برنده شد",
        "ai_wins": "بازیکن ۲ برنده شد",
        "reset_scores_warning": "آیا مطمئن هستید که می‌خواهید امتیازات را بازنشانی کنید؟",
        "pieces": "مهره‌ها",
        "footer": "",
        "confirm": "تأیید",
        "new_game_warning": "آیا بازی جدیدی شروع شود؟",
        "warning": "هشدار",
        "game_settings_tab": "بازی",
        "design_tab": "طراحی",
        "ai_tab": "هوش مصنوعی",
        "player_tab": "بازیکنان",
        "play_with": "بازی با",
        "human_vs_human": "انسان در برابر انسان",
        "human_vs_ai": "انسان در برابر هوش مصنوعی",
        "ai_vs_ai": "هوش مصنوعی در برابر هوش مصنوعی",
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
        "ai_1_name": "نام هوش مصنوعی ۱",
        "ai_2_name": "نام هوش مصنوعی ۲",
        "upload_image": "بارگذاری تصویر",
        "save_changes": "ذخیره",
        "close": "بستن",
        "reset_settings": "بازنشانی",
        "coming_soon": "به زودی",
        "player_1_ai_type": "نوع هوش مصنوعی بازیکن ۱",
        "player_2_ai_type": "نوع هوش مصنوعی بازیکن ۲",
        "error": "خطا",
        "ai_pause_error": "زمان مکث هوش مصنوعی باید بین ۰ تا ۵۰۰۰ میلی‌ثانیه باشد",
        "invalid_number_hands": "تعداد دست‌ها باید بین ۱ تا ۱۰۰۰ باشد",
        "invalid_number_error": "لطفاً اعداد معتبر وارد کنید",
        "apply_after_game": "تنظیمات پس از پایان بازی اعمال خواهند شد",
        "reset_settings_warning": "آیا مطمئن هستید که می‌خواهید تمام تنظیمات را بازنشانی کنید؟",
        # متن‌های جدید برای تب Advanced Settings
        "advanced_settings_tab": "تنظیمات پیشرفته",
        "advanced_settings_warning": "این تنظیمات ممکن است روی کارایی هوش‌های مصنوعی تأثیر بگذارد. در صورتی که در این زمینه آگاهی ندارید، پیشنهاد می‌کنیم از گزینه‌های پیش‌فرض استفاده کنید.",
        "ai_ability_level": "سطح توانایی هوش مصنوعی",
        "very_weak": "خیلی ضعیف",
        "weak": "ضعیف",
        "medium": "متوسط",
        "strong": "قوی",
        "very_strong": "خیلی قوی",
        "advanced_config": "پیکربندی پیشرفته",
        "ai_1_training_params": "پارامترهای آموزش هوش مصنوعی ۱",
        "ai_2_training_params": "پارامترهای آموزش هوش مصنوعی ۲",
        "az1_training_params": "پارامترهای آموزش AlphaZero ۱",
        "az2_training_params": "پارامترهای آموزش AlphaZero ۲",
        "mcts_params": "پارامترهای MCTS",
        "network_params": "پارامترهای شبکه",
        "advanced_nn_params": "پارامترهای شبکه پیشرفته",
        "reset_tab": "بازنشانی این تب",
        "save_all": "ذخیره همه",
        "invalid_input": "ورودی نامعتبر. لطفاً مقدار معتبر وارد کنید."
    }
}

def get_config_path():
    """بازگرداندن مسیر فایل config.json"""
    return Path(__file__).parent.parent / "config.json"

def get_stats_path():
    """بازگرداندن مسیر فایل stats.json"""
    return Path(__file__).parent.parent / "stats.json"

def load_config():
    """بارگذاری تنظیمات از فایل config.json"""
    default_config = {
        # تنظیمات رابط کاربری (منتقل شده از constants.py)
        "square_size": 80,
        "board_size": 8,
        "border_thickness": 7,
        "menu_height": 30,
        "window_width": 940,  # محاسبه شده: BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 2 + 300
        "board_width": 640,   # محاسبه شده: BOARD_SIZE * SQUARE_SIZE
        "window_height": 727, # محاسبه شده: BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 3 + 60
        "panel_width": 300,
        "button_spacing_from_bottom": 40,
        "animation_frames": 1,
        "player_image_size": 75,  # محاسبه شده: int(PANEL_WIDTH * 0.25)
        "config_file": "settings.json",
        "stats_file": "game_stats.json",
        "game_version": "1.0",
        # تنظیمات بازی
        "player_starts": True,
        "use_timer": True,
        "game_time": 5,
        "language": "en",
        "player_1_color": "#ff0000",
        "player_2_color": "#0000ff",
        "board_color_1": "#ffffff",
        "board_color_2": "#8b4513",
        "piece_style": "circle",
        "sound_enabled": False,
        "ai_pause_time": 20,
        "az1_ability": 5,
        "az2_ability": 5,
        "ai_1_ability": 5,
        "ai_2_ability": 5,
        "game_mode": "human_vs_human",
        "ai_vs_ai_mode": "only_once",
        "repeat_hands": 10,
        "player_1_name": "Player 1",
        "player_2_name": "Player 2",
        "ai_1_name": "AI 1",
        "ai_2_name": "AI 2",
        "player_1_image": "",
        "player_2_image": "",
        "ai_1_image": "",
        "ai_2_image": "",
        "player_1_piece_image": "",
        "player_1_king_image": "",
        "player_2_piece_image": "",
        "player_2_king_image": "",
        "pause_between_hands": 1000,
        "player_1_ai_type": "none",
        "player_2_ai_type": "advanced",
        # پارامترهای هوش مصنوعی
        "ai_1_training_params": {
            "memory_size": 10000,
            "batch_size": 128,
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "update_target_every": 5,
            "reward_threshold": 5.0,
            "gradient_clip": 1.0,
            "target_update_alpha": 0.01
        },
        "ai_2_training_params": {
            "memory_size": 10000,
            "batch_size": 128,
            "learning_rate": 0.0005,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "update_target_every": 5,
            "reward_threshold": 5.0,
            "gradient_clip": 1.0,
            "target_update_alpha": 0.01
        },
        "az1_training_params": {
            "memory_size": 10000,
            "batch_size": 16,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "weight_decay": 1e-4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "update_target_every": 100,
            "replay_batch_size": 32
        },
        "az2_training_params": {
            "memory_size": 10000,
            "batch_size": 16,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "weight_decay": 1e-4,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.995,
            "update_target_every": 100,
            "replay_batch_size": 32
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
        }
    }
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                default_config.update(config)
                print(f"Loaded config from {config_path}")
        else:
            print(f"Config file not found at {config_path}, using default config")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading config from {config_path}: {e}, using default config")
    return default_config

def save_config(config):
    """ذخیره تنظیمات در فایل config.json"""
    config_path = get_config_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")

def load_stats():
    """بارگذاری آمار بازی"""
    stats_path = get_stats_path()
    default_stats = {
        "player_1_wins": 0,
        "player_2_wins": 0,
        "ai_stats": {
            "ai_1": {"wins": 0, "losses": 0, "draws": 0, "move_times": []},
            "ai_2": {"wins": 0, "losses": 0, "draws": 0, "move_times": []}
        }
    }
    try:
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                default_stats.update(stats)
                print(f"Loaded stats from {stats_path}")
        else:
            print(f"Stats file not found at {stats_path}, using default stats")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading stats from {stats_path}: {e}, using default stats")
    return default_stats

def save_stats(stats):
    """ذخیره آمار بازی"""
    stats_path = get_stats_path()
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"Stats saved to {stats_path}")
    except Exception as e:
        print(f"Error saving stats to {stats_path}: {e}")


        #ادغام تمام کلیدهای LANGUAGES از constants.py با نسخه قبلی config.py.

#اضافه کردن ثابت‌های رابط کاربری (square_size, board_size, و غیره) به default_config.

#انتقال config_file, stats_file, و game_version به default_config.

#به‌روزرسانی مسیرهای get_config_path و get_stats_path برای سازگاری با ساختار پروژه.

