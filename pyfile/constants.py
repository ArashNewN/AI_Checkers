#constants.py

SQUARE_SIZE = 80
BOARD_SIZE = 8
BORDER_THICKNESS = 7
MENU_HEIGHT = 30
WINDOW_WIDTH = BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 2 + 410
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
WINDOW_HEIGHT = BOARD_SIZE * SQUARE_SIZE + BORDER_THICKNESS * 3 + 10
PANEL_WIDTH = 300
BUTTON_SPACING_FROM_BOTTOM = 40
ANIMATION_FRAMES = 15
PLAYER_IMAGE_SIZE = int(PANEL_WIDTH * 0.25)
GAME_VERSION = "1.0"

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)
SKY_BLUE = (135, 206, 235)
LIGHT_GRAY = (211, 211, 211)
LIGHT_GREEN = (144, 238, 144)

CONFIG_FILE = 'settings.json'
STATS_FILE = 'game_stats.json'

LANGUAGES = {
    "en": {
        "learning_rate": "Learning Rate",
        "gamma": "Gamma",
        "batch_size": "Batch Size",
        "memory_size": "Memory Size",
        "cache_save_interval": "Cache Save Interval",
        "cache_file": "Cache File",
        "piece_difference": "Piece Difference",
        "king_bonus": "King Bonus",
        "position_bonus": "Position Bonus",
        "capture_bonus": "Capture Bonus",
        "multi_jump_bonus": "Multi-Jump Bonus",
        "king_capture_bonus": "King Capture Bonus",
        "mobility_bonus": "Mobility Bonus",
        "safety_penalty": "Safety Penalty",
        "piece_images": "Piece Images",
        "player_1_piece": "Player 1 Piece",
        "player_1_king": "Player 1 King",
        "player_2_piece": "Player 2 Piece",
        "player_2_king": "Player 2 King",
        "ai_progress": "AI Progress",
        "az_progress": "AZ Progress",
        "wins": "Wins",
        "ai_code": "ai_code",
        "losses": "Losses",
        "add_new_ai": "add_new_ai",
        "draws": "Draws",
        "time": "TIME:",
        "pause": "pause:",
        "resume": "resume:",
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
        "ai_1_name": "AI 1",
        "ai_2_name": "AI 2",
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
        "ai_settings_tab": "AI",
        "add_ai": "add_ai",
        "new_ai_type": "new_ai_type",
        "strong": "Strong",
        "very_strong": "Very Strong",
        "select_module_class": "select_module_class",
        "advanced_config": "Advanced Config",
        "open_ai_progress": "open_ai_progress",
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
        "confirm_reset_all": "confirm_reset_all",
        "confirm_reset": "confirm_resetl",
        "reset": "reset",
        "error_saving_settings": "Failed to save settings: {error}",
        "model_not_found": "Model file not found",
        "model_load_error": "Failed to load model: {error}",
        "training_progress": "Training Progress",
        "model_parameters": "Model Parameters",
        "parameter": "Parameter",
        "shape": "Shape",
        "num_elements": "Num Elements",
        "total_parameters": "Total Parameters",
        "ability": "Ability",
        "ai_pause_time_ms": "AI Pause Time (ms)",
        "player_1_ai": "Player 1 AI",
        "player_2_ai": "Player 2 AI",
        "select_ai_to_remove": "Please select an AI to remove.",
        "ai_players": "ai_players",
        "no_ai_selected": "No AI selected",
        "sound_enabled": "sound_enabled",
        "hint_on": "Hint On",
        "hint_off": "Hint Off",
        "undo": "Undo",
        "redo": "Redo",
        "invalid_input": "Invalid input. Please enter a valid value.",
    },
    "fa": {
        "learning_rate": "نرخ یادگیری",
        "gamma": "گاما",
        "batch_size": "اندازه دسته",
        "memory_size": "اندازه حافظه",
        "cache_save_interval": "فاصله ذخیره کش",
        "cache_file": "فایل کش",
        "piece_difference": "تفاوت مهره",
        "ai_1_name": "AI 1",
        "ai_2_name": "AI 2",
        "king_bonus": "امتیاز شاه",
        "position_bonus": "امتیاز موقعیت",
        "capture_bonus": "امتیاز گرفتن",
        "multi_jump_bonus": "امتیاز پرش چندگانه",
        "king_capture_bonus": "امتیاز گرفتن شاه",
        "mobility_bonus": "امتیاز تحرک",
        "safety_penalty": "جریمه ایمنی",
        "piece_images": "تصاویر مهره‌ها",
        "sound_enabled": "sound_enabled",
        "hint_on": "راهنمایی روشن",
        "hint_off": "راهنمایی خاموش",
        "undo": "بازگشت",
        "redo": "بازگرداندن",
        "confirm_reset": "confirm_resetl",
        "player_1_piece": "مهره بازیکن ۱",
        "player_1_king": "شاه بازیکن ۱",
        "info": "info",
        "no_ai_selected": "No AI selected",
        "player_2_piece": "مهره بازیکن ۲",
        "confirm_reset_all": "confirm_reset_all",
        "open_ai_progress": "open_ai_progress",
        "player_2_king": "شاه بازیکن ۲",
        "add_ai": "add_ai",
        "ability": "ability",
        "pause": "pause:",
        "resume": "resume:",
        "status": "status",
        "ai_players": "ai_players",
        "error_saving_settings": "خطا در ذخیره تنظیمات: {error}",
        "model_not_found": "فایل مدل پیدا نشد",
        "model_load_error": "خطا در بارگذاری مدل: {error}",
        "training_progress": "پیشرفت آموزش",
        "model_parameters": "پارامترهای مدل",
        "parameter": "پارامتر",
        "shape": "شکل",
        "num_elements": "تعداد عناصر",
        "total_parameters": "مجموع پارامترها",
        "player_1_ai": "player_1_ai",
        "player_2_ai": "player_2_ai",
        "ai_pause_time_ms": "ai_pause_time_ms",
        "player_1_color": "player_1_color",
        "player_2_color": "player_2_color",
        "add": "add",
        "player_2_ability": "player_2_ability",
        "player_1_ability": "player_1_ability",
        "remove": "remove",
        "ai_progress": "پیشرفت هوش مصنوعی",
        "remove_selected": "remove_selected",
        "az_progress": "پیشرفت هوش مصنوعی",
        "ai_type": "ai_type",
        "search_modules": "search_modules",
        "description": "description",
        "add_new_ai": "add_new_ai",
        "settings_reset": "تنظیمات به مقادیر پیش‌فرض بازنشانی شدند.",
        "ai_added": "هوش مصنوعی با موفقیت اضافه شد.",
        "ai_settings_tab": "AI",
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
        "ai_code": "ai_code",
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
        "reset": "reset",
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
        "ai_1_training_params": "پارامترهای آموزش  AI ۱",
        "ai_2_training_params": "پارامترهای آموزش  AI ۲",
        "mcts_params": "پارامترهای MCTS",
        "select_module_class": "select_module_class",
        "network_params": "پارامترهای شبکه",
        "advanced_nn_params": "پارامترهای شبکه پیشرفته",
        "reset_tab": "بازنشانی این تب",
        "save_all": "ذخیره همه",
        "invalid_input": "ورودی نامعتبر. لطفاً مقدار معتبر وارد کنید."
    }
}