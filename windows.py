#windows.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from typing import Literal
from .settings import GameSettings
from .constants import LANGUAGES
from .config import save_config, load_config, load_ai_config, save_ai_config, load_ai_specific_config, \
    save_ai_specific_config, DEFAULT_AI_PARAMS
from .utils import hex_to_rgb, rgb_to_hex
import torch
import json
import sys
import importlib
from pathlib import Path
import logging

# ایجاد پوشه logs در صورت عدم وجود
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.DEBUG,
    filename=Path(__file__).parent.parent / "logs" / "app.log",
    encoding="utf-8",
    format=" %(levelname)s - %(message)s"
)

# مسیر پروژه
project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))


def configure_styles():
    style = ttk.Style()
    style.configure("Custom.TButton", font=("Arial", 10), padding=5, background="#4CAF50", foreground="black")
    style.map("Custom.TButton", background=[("active", "#45a049")], foreground=[("active", "black")])
    style.configure("TLabel", font=("Arial", 10))
    style.configure("TEntry", font=("Arial", 10))
    style.configure("TCombobox", font=("Arial", 10))


class BaseWindow:
    def __init__(self, interface, root=None):
        self.interface = interface
        self.settings = interface.settings if interface else None
        self.root = root
        self.window = None
        self.tk = None  # تعریف tk در __init__
        self.is_open = False
        self.config = load_config()
        self.project_dir = Path(__file__).parent.parent
        self.pth_dir = self.project_dir / "pth"
        configure_styles()

    def create_window(self, title, width=None, height=None):
        if self.is_open or self.window:
            return
        self.window = tk.Toplevel(self.root) if self.root else tk.Toplevel()
        self.tk = self.window
        self.window.title(title)
        width = width or self.config.get("settings_window_width", 400)
        height = height or self.config.get("settings_window_height", 300)
        x = (self.config['window_width'] - width) // 2
        y = (self.config['window_height'] - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.minsize(self.config.get("min_window_width", 300), self.config.get("min_window_height", 200))
        self.is_open = True
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def update(self):
        if self.window and self.is_open:
            try:
                self.window.update()
            except tk.TclError:
                self.close()

    def close(self):
        if self.window and self.is_open:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            self.window = None
            self.is_open = False
            if self.root:
                try:
                    self.root.update()
                except tk.TclError:
                    pass

    def log_error(self, message):
        logging.error(message)
        if self.window and self.is_open:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                message,
                parent=self.window
            )


class SettingsWindow(BaseWindow):
    def __init__(self, interface, root=None):
        super().__init__(interface, root)
        self.temp_settings = GameSettings()
        self.copy_current_settings()
        self.ai_types = ["none"]
        self.entries = {}
        self.load_ai_config()
        # تعریف متغیرهای نمونه‌ای در __init__
        self.ai_list = None
        self.ai_pause_var = tk.IntVar(value=self.temp_settings.ai_pause_time)
        self.ai_vs_ai_subframe = None
        self.ai_vs_ai_var = tk.StringVar(value=self.temp_settings.ai_vs_ai_mode)
        self.al1_name_var = tk.StringVar(value=self.temp_settings.al1_name)
        self.al2_name_var = tk.StringVar(value=self.temp_settings.al2_name)
        self.b1_color_button = None
        self.b2_color_button = None
        self.board_color_1_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_1))
        self.board_color_2_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_2))
        self.hmh_entry = None
        self.hmh_var = tk.IntVar(value=self.temp_settings.repeat_hands)
        self.lang_var = tk.StringVar(value=self.temp_settings.language)
        self.p1_color_button = None
        self.p2_color_button = None
        self.piece_style_var = tk.StringVar(value=self.temp_settings.piece_style)
        self.play_with_var = tk.StringVar(value=self.temp_settings.game_mode)
        self.player_1_ability_menu = None
        self.player_1_ability_var = tk.StringVar(value="medium")
        self.player_1_ai_menu = None
        self.player_1_ai_type_var = tk.StringVar(value="none")
        self.player_1_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_1_color))
        self.player_1_name_var = tk.StringVar(value=self.temp_settings.player_1_name)
        self.player_2_ability_menu = None
        self.player_2_ability_var = tk.StringVar(value="medium")
        self.player_2_ai_menu = None
        self.player_2_ai_type_var = tk.StringVar(value="none")
        self.player_2_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_2_color))
        self.player_2_name_var = tk.StringVar(value=self.temp_settings.player_2_name)
        self.preview_canvas = None
        self.repeat_hands_frame = None
        self.repeat_once_rb = None
        self.repeat_until_rb = None
        self.sound_var = tk.BooleanVar(value=self.temp_settings.sound_enabled)
        self.start_var = tk.StringVar(value="player_1" if self.temp_settings.player_starts else "player_2")
        self.timer_combo = None
        self.timer_duration_frame = None
        self.timer_var = tk.StringVar(value="with_timer" if self.temp_settings.use_timer else "no_timer")
        self.timer_var_duration = tk.IntVar(value=self.temp_settings.game_time)
        # متغیرهای تنظیمات پیشرفته
        self.player1_learning_rate_var = tk.DoubleVar(value=0.001)
        self.player1_gamma_var = tk.DoubleVar(value=0.99)
        self.player1_batch_size_var = tk.IntVar(value=64)
        self.player1_memory_size_var = tk.IntVar(value=10000)
        self.player1_piece_difference_var = tk.DoubleVar(value=1.0)
        self.player1_king_bonus_var = tk.DoubleVar(value=2.0)
        self.player1_position_bonus_var = tk.DoubleVar(value=0.1)
        self.player1_capture_bonus_var = tk.DoubleVar(value=1.5)
        self.player1_multi_jump_bonus_var = tk.DoubleVar(value=0.5)
        self.player1_king_capture_bonus_var = tk.DoubleVar(value=2.5)
        self.player1_mobility_bonus_var = tk.DoubleVar(value=0.2)
        self.player1_safety_penalty_var = tk.DoubleVar(value=-0.1)
        self.player2_learning_rate_var = tk.DoubleVar(value=0.001)
        self.player2_gamma_var = tk.DoubleVar(value=0.99)
        self.player2_batch_size_var = tk.IntVar(value=64)
        self.player2_memory_size_var = tk.IntVar(value=10000)
        self.player2_piece_difference_var = tk.DoubleVar(value=1.0)
        self.player2_king_bonus_var = tk.DoubleVar(value=2.0)
        self.player2_position_bonus_var = tk.DoubleVar(value=0.1)
        self.player2_capture_bonus_var = tk.DoubleVar(value=1.5)
        self.player2_multi_jump_bonus_var = tk.DoubleVar(value=0.5)
        self.player2_king_capture_bonus_var = tk.DoubleVar(value=2.5)
        self.player2_mobility_bonus_var = tk.DoubleVar(value=0.2)
        self.player2_safety_penalty_var = tk.DoubleVar(value=-0.1)

    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["settings"],
                           self.config.get("settings_window_width", 500),
                           self.config.get("settings_window_height", 750))
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True)

        self.setup_game_tab(notebook)
        self.setup_design_tab(notebook)
        self.setup_ai_tab(notebook)
        self.setup_player_tab(notebook)
        self.setup_advanced_tab(notebook)

        button_frame = ttk.Frame(self.window, padding=10)
        button_frame.pack(fill="x")

        side: Literal["left", "right"] = "right" if self.settings.language == "fa" else "left"
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["save_changes"],
                   command=self.save, style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["reset_settings"],
                   command=self.reset, style="Custom.TButton").pack(side=side, padx=5)

        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.update_ai_dropdowns()

    def load_ai_config(self):
        ai_config = load_ai_config()
        self.ai_types = ["none"] + list(ai_config.get("ai_types", {}).keys())
        self.temp_settings.ai_configs = ai_config.get("ai_configs", {
            "player_1": {"ai_type": "none", "ai_code": None, "ability_level": 5, "params": {}},
            "player_2": {"ai_type": "none", "ai_code": None, "ability_level": 5, "params": {}}
        })

    @staticmethod
    def check_ai_module(ai_type: str) -> bool:
        if ai_type == "none":
            return True
        ai_config = load_ai_config()
        print(f"check_ai_module: Checking ai_type '{ai_type}' in {ai_config['ai_types'].keys()}")
        return ai_type in ai_config["ai_types"]

    def update_ai_dropdowns(self):
        ai_config = load_ai_config()
        print(f"Updating AI dropdowns with ai_types: {ai_config['ai_types'].keys()}")

        self.player_1_ai_menu['menu'].delete(0, 'end')
        self.player_2_ai_menu['menu'].delete(0, 'end')
        for ai_type in self.ai_types:
            self.player_1_ai_menu['menu'].add_command(
                label=ai_type,
                command=lambda value=ai_type: [
                    self.player_1_ai_type_var.set(value),
                    self._update_player_ai_config("player_1", value)
                ]
            )
            self.player_2_ai_menu['menu'].add_command(
                label=ai_type,
                command=lambda value=ai_type: [
                    self.player_2_ai_type_var.set(value),
                    self._update_player_ai_config("player_2", value)
                ]
            )

        if self.player_1_ai_type_var.get() not in self.ai_types:
            self.player_1_ai_type_var.set("none")
            self._update_player_ai_config("player_1", "none")
            print(f"Reset player_1 AI to 'none' (was {self.player_1_ai_type_var.get()})")
        if self.player_2_ai_type_var.get() not in self.ai_types:
            self.player_2_ai_type_var.set("none")
            self._update_player_ai_config("player_2", "none")
            print(f"Reset player_2 AI to 'none' (was {self.player_2_ai_type_var.get()})")

    def _update_player_ai_config(self, player: str, ai_type: str):
        ai_config = load_ai_config()
        print(f"Updating {player} temp_settings to AI: {ai_type}")

        if ai_type != "none" and ai_type not in ai_config["ai_types"]:
            print(f"Error: AI type '{ai_type}' not found in ai_config['ai_types']")
            self.log_error(f"ماژول AI {ai_type} برای {player} پیدا نشد")
            self.player_1_ai_type_var.set("none") if player == "player_1" else self.player_2_ai_type_var.set("none")
            ai_type = "none"

        self.temp_settings.ai_configs[player]["ai_type"] = ai_type
        if ai_type != "none":
            code = ai_config["ai_types"][ai_type]["code"]
            self.temp_settings.ai_configs[player]["ai_code"] = code
            ai_specific_config = load_ai_specific_config(code)
            self.temp_settings.ai_configs[player]["params"] = ai_specific_config.get(player, DEFAULT_AI_PARAMS.copy())
            self.temp_settings.ai_configs[player]["ability_level"] = 5
        else:
            self.temp_settings.ai_configs[player]["ai_code"] = None
            self.temp_settings.ai_configs[player]["params"] = {}
            self.temp_settings.ai_configs[player]["ability_level"] = 5

        print(f"Updated {player} temp_settings: {self.temp_settings.ai_configs[player]}")

    def search_ai_modules(self):
        from a.base_ai import BaseAI
        config = load_config()
        ai_dirs = config.get("ai_module_dirs", ["a"])
        ai_modules = []

        for ai_dir in ai_dirs:
            module_dir = self.project_dir / ai_dir
            for py_file in module_dir.glob("*.py"):
                module_name = py_file.stem
                if module_name in ["__init__", "windows", "base_ai"]:
                    print(f"Skipping file: {module_name}")
                    continue
                try:
                    module = importlib.import_module(f"{ai_dir}.{module_name}")
                    print(f"Loaded module: {ai_dir}.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseAI) and attr != BaseAI:
                            metadata = getattr(attr, "metadata", None)
                            if metadata and all(key in metadata for key in ["type", "description", "code"]):
                                ai_modules.append({
                                    "display": module_name,
                                    "default_type": metadata["type"],
                                    "default_description": metadata["description"],
                                    "default_code": metadata["code"],
                                    "class_name": attr_name
                                })
                                print(f"Found AI: {module_name:}.{attr_name}, metadata: {metadata}")
                            else:
                                print(f"Skipped {module_name}.{attr_name}: Invalid metadata")
                except Exception as e:
                    print(f"Error loading {ai_dir}.{module_name}: {str(e)}")

        ai_config = load_ai_config()
        for module in ai_modules:
            ai_type = module["default_type"]
            code = module["default_code"]
            used_codes = {info.get("code") for info in ai_config["ai_types"].values()}
            if ai_type not in ai_config["ai_types"] and code not in used_codes:
                ai_config["ai_types"][ai_type] = {
                    "module": f"{ai_dir}.{module['display']}",
                    "class": module["class_name"],
                    "description": module["default_description"],
                    "code": code
                }
                save_ai_specific_config(code, {
                    "player_1": DEFAULT_AI_PARAMS.copy(),
                    "player_2": DEFAULT_AI_PARAMS.copy()
                })
        save_ai_config(ai_config)
        self.ai_types = ["none"] + list(ai_config["ai_types"].keys())
        self.update_ai_dropdowns()

    def setup_ai_tab(self, notebook):
        ai_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ai_frame, text=LANGUAGES[self.settings.language]["ai_settings_tab"])

        self.ai_list = ttk.Treeview(ai_frame, columns=("Type", "Code", "Description"), show="headings", height=4)
        self.ai_list.heading("Type", text=LANGUAGES[self.settings.language]["ai_type"])
        self.ai_list.heading("Code", text=LANGUAGES[self.settings.language]["ai_code"])
        self.ai_list.heading("Description", text=LANGUAGES[self.settings.language]["description"])
        self.ai_list.pack(fill="both", expand=False, pady=5)

        ai_players_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["ai_players"], padding=10)
        ai_players_frame.pack(fill="x", pady=5)

        player_1_container = ttk.Frame(ai_players_frame)
        player_1_container.pack(fill="x", pady=2)
        ttk.Label(player_1_container, text=LANGUAGES[self.settings.language]["player_1_ai"]).pack(side="left", padx=5)
        player_1_ai_type = self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none")
        if player_1_ai_type not in self.ai_types:
            player_1_ai_type = "none"
        self.player_1_ai_type_var.set(player_1_ai_type)
        self.player_1_ai_menu = ttk.OptionMenu(player_1_container, self.player_1_ai_type_var,
                                               player_1_ai_type, *self.ai_types)
        self.player_1_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_1_ai_menu["menu"].config(font=("Arial", 10))
        self.player_1_ai_type_var.trace("w", lambda *args: self._update_player_ai_config("player_1",
                                                                                         self.player_1_ai_type_var.get()))

        ttk.Label(player_1_container, text=LANGUAGES[self.settings.language]["ability"]).pack(side="left", padx=5)
        ability_mapping = {
            1: "very_weak",
            3: "weak",
            5: "medium",
            7: "strong",
            9: "very_strong"
        }
        player_1_ability = self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5)
        self.player_1_ability_var.set(
            LANGUAGES[self.settings.language][ability_mapping.get(player_1_ability, "medium")])
        self.player_1_ability_menu = ttk.OptionMenu(
            player_1_container, self.player_1_ability_var, self.player_1_ability_var.get(),
            LANGUAGES[self.settings.language]["very_weak"],
            LANGUAGES[self.settings.language]["weak"],
            LANGUAGES[self.settings.language]["medium"],
            LANGUAGES[self.settings.language]["strong"],
            LANGUAGES[self.settings.language]["very_strong"],
            command=lambda _: self.update_ability_levels("player_1")
        )
        self.player_1_ability_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_1_ability_menu["menu"].config(font=("Arial", 10))

        player_2_container = ttk.Frame(ai_players_frame)
        player_2_container.pack(fill="x", pady=2)
        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["player_2_ai"]).pack(side="left", padx=5)
        player_2_ai_type = self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none")
        if player_2_ai_type not in self.ai_types:
            player_2_ai_type = "none"
        self.player_2_ai_type_var.set(player_2_ai_type)
        self.player_2_ai_menu = ttk.OptionMenu(player_2_container, self.player_2_ai_type_var,
                                               player_2_ai_type, *self.ai_types)
        self.player_2_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_2_ai_menu["menu"].config(font=("Arial", 10))
        self.player_2_ai_type_var.trace("w", lambda *args: self._update_player_ai_config("player_2",
                                                                                         self.player_2_ai_type_var.get()))

        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["ability"]).pack(side="left", padx=5)
        player_2_ability = self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5)
        self.player_2_ability_var.set(
            LANGUAGES[self.settings.language][ability_mapping.get(player_2_ability, "medium")])
        self.player_2_ability_menu = ttk.OptionMenu(
            player_2_container, self.player_2_ability_var, self.player_2_ability_var.get(),
            LANGUAGES[self.settings.language]["very_weak"],
            LANGUAGES[self.settings.language]["weak"],
            LANGUAGES[self.settings.language]["medium"],
            LANGUAGES[self.settings.language]["strong"],
            LANGUAGES[self.settings.language]["very_strong"],
            command=lambda _: self.update_ability_levels("player_2")
        )
        self.player_2_ability_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_2_ability_menu["menu"].config(font=("Arial", 10))

        pause_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["ai_pause_time"], padding=10)
        pause_frame.pack(fill="x", pady=5)
        ttk.Label(pause_frame, text=LANGUAGES[self.settings.language]["ai_pause_time_ms"]).pack(anchor="w")
        self.ai_pause_var.trace("w", lambda *args: self.update_temp_settings("ai_pause_time", self.ai_pause_var.get()))
        ttk.Entry(pause_frame, textvariable=self.ai_pause_var, width=10).pack(anchor="w", pady=5)

        self.update_ai_list()
        self.search_ai_modules()

    def update_ai_list(self):
        self.ai_list.delete(*self.ai_list.get_children())
        ai_config = load_ai_config()
        for ai_type, ai_info in ai_config["ai_types"].items():
            description = ai_info.get("description", "")
            code = ai_info.get("code", "")
            self.ai_list.insert("", "end", values=(ai_type, code, description))

    def copy_current_settings(self):
        config = load_config()
        ai_config = load_ai_config()
        for key, value in config.items():
            if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                value = hex_to_rgb(value)
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
            if hasattr(self.temp_settings, key):
                setattr(self.temp_settings, key, value)
        self.temp_settings.ai_configs = ai_config.get("ai_configs", {
            "player_1": {"ai_type": "none", "ability_level": 5, "training_params": {}, "reward_weights": {}},
            "player_2": {"ai_type": "none", "ability_level": 5, "training_params": {}, "reward_weights": {}}
        })

    def update_ability_levels(self, player: str):
        ability_levels = {
            LANGUAGES[self.settings.language]["very_weak"]: 1,
            LANGUAGES[self.settings.language]["weak"]: 3,
            LANGUAGES[self.settings.language]["medium"]: 5,
            LANGUAGES[self.settings.language]["strong"]: 7,
            LANGUAGES[self.settings.language]["very_strong"]: 9
        }
        if player == "player_1":
            selected_level = self.player_1_ability_var.get()
            self.temp_settings.ai_configs["player_1"]["ability_level"] = ability_levels.get(selected_level, 5)
        elif player == "player_2":
            selected_level = self.player_2_ability_var.get()
            self.temp_settings.ai_configs["player_2"]["ability_level"] = ability_levels.get(selected_level, 5)

    def update_temp_settings(self, key: str, value):
        if key == "language":
            self.temp_settings.language = value
        elif key == "game_mode":
            self.temp_settings.game_mode = value
            self.toggle_ai_vs_ai_options()
        elif key == "ai_vs_ai_mode":
            self.temp_settings.ai_vs_ai_mode = value
            self.toggle_repeat_options()
        elif key == "repeat_hands":
            self.temp_settings.repeat_hands = value
        elif key == "player_starts":
            self.temp_settings.player_starts = value
        elif key == "use_timer":
            self.temp_settings.use_timer = value
            self.toggle_timer()
        elif key == "game_time":
            self.temp_settings.game_time = value
        elif key == "piece_style":
            self.temp_settings.piece_style = value
        elif key == "player_1_color":
            self.temp_settings.player_1_color = value
        elif key == "player_2_color":
            self.temp_settings.player_2_color = value
        elif key == "board_color_1":
            self.temp_settings.board_color_1 = value
        elif key == "board_color_2":
            self.temp_settings.board_color_2 = value
        elif key == "sound_enabled":
            self.temp_settings.sound_enabled = value
        elif key == "player_1_ai_type":
            self.temp_settings.player_1_ai_type = value
            self.temp_settings.ai_configs["player_1"] = {
                "ai_type": value,
                "ai_code": None,
                "ability_level": self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5),
                "params": {}
            }
            if value != "none":
                ai_config = load_ai_config()
                if value in ai_config["ai_types"]:
                    self.temp_settings.ai_configs["player_1"]["ai_code"] = ai_config["ai_types"][value]["code"]
                    ai_specific_config = load_ai_specific_config(self.temp_settings.ai_configs["player_1"]["ai_code"])
                    self.temp_settings.ai_configs["player_1"]["params"] = ai_specific_config.get("player_1",
                                                                                                 DEFAULT_AI_PARAMS.copy())
        elif key == "player_2_ai_type":
            self.temp_settings.player_2_ai_type = value
            self.temp_settings.ai_configs["player_2"] = {
                "ai_type": value,
                "ai_code": None,
                "ability_level": self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5),
                "params": {}
            }
            if value != "none":
                ai_config = load_ai_config()
                if value in ai_config["ai_types"]:
                    self.temp_settings.ai_configs["player_2"]["ai_code"] = ai_config["ai_types"][value]["code"]
                    ai_specific_config = load_ai_specific_config(self.temp_settings.ai_configs["player_2"]["ai_code"])
                    self.temp_settings.ai_configs["player_2"]["params"] = ai_specific_config.get("player_2",
                                                                                                 DEFAULT_AI_PARAMS.copy())
        elif key == "ai_pause_time":
            self.temp_settings.ai_pause_time = value

    def setup_game_tab(self, notebook):
        game_frame = ttk.Frame(notebook, padding="10")
        notebook.add(game_frame, text=LANGUAGES[self.settings.language]["game_settings_tab"])

        lang_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["language"], padding=10)
        lang_frame.pack(fill="x", pady=5)
        self.lang_var.trace("w", lambda *args: self.update_temp_settings("language", self.lang_var.get()))
        lang_menu = ttk.OptionMenu(lang_frame, self.lang_var, self.temp_settings.language, "en", "fa")
        lang_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        lang_frame.grid_columnconfigure(0, weight=1)
        lang_menu["menu"].config(font=("Arial", 10))

        play_with_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["play_with"], padding=10)
        play_with_frame.pack(fill="x", pady=5)
        self.play_with_var.trace("w", lambda *args: self.update_temp_settings("game_mode", self.play_with_var.get()))
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["human_vs_human"],
                        variable=self.play_with_var, value="human_vs_human",
                        command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["human_vs_ai"],
                        variable=self.play_with_var, value="human_vs_ai",
                        command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["ai_vs_ai"],
                        variable=self.play_with_var, value="ai_vs_ai",
                        command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)

        self.ai_vs_ai_subframe = ttk.Frame(game_frame, padding=10)
        self.ai_vs_ai_subframe.pack(fill="x", pady=5)
        self.ai_vs_ai_var.trace("w", lambda *args: self.update_temp_settings("ai_vs_ai_mode", self.ai_vs_ai_var.get()))
        self.repeat_once_rb = ttk.Radiobutton(self.ai_vs_ai_subframe,
                                              text=LANGUAGES[self.settings.language]["only_once"],
                                              variable=self.ai_vs_ai_var, value="only_once",
                                              command=self.toggle_repeat_options)
        self.repeat_once_rb.pack(side="left", padx=10)
        self.repeat_until_rb = ttk.Radiobutton(self.ai_vs_ai_subframe,
                                               text=LANGUAGES[self.settings.language]["repeat_game"],
                                               variable=self.ai_vs_ai_var, value="repeat_game",
                                               command=self.toggle_repeat_options)
        self.repeat_until_rb.pack(side="left", padx=10)

        if self.play_with_var.get() != "ai_vs_ai":
            self.repeat_once_rb.config(state="disabled")
            self.repeat_until_rb.config(state="disabled")
        else:
            self.repeat_once_rb.config(state="normal")
            self.repeat_until_rb.config(state="normal")

        self.repeat_hands_frame = ttk.Frame(self.ai_vs_ai_subframe)
        self.hmh_var.trace("w", lambda *args: self.update_temp_settings("repeat_hands", self.hmh_var.get()))
        self.hmh_entry = ttk.Entry(self.repeat_hands_frame, textvariable=self.hmh_var, width=5)
        self.hmh_entry.pack(side="left", padx=5)
        self.toggle_repeat_options()

        start_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["starting_player"], padding=10)
        start_frame.pack(fill="x", pady=5)
        self.start_var.trace("w", lambda *args: self.update_temp_settings("player_starts",
                                                                          self.start_var.get() == "player_1"))
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["player"],
                        variable=self.start_var, value="player_1").pack(side="left", padx=10)
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["ai"],
                        variable=self.start_var, value="player_2").pack(side="left", padx=10)

        timer_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["game_timer"], padding=10)
        timer_frame.pack(fill="x", pady=5)
        self.timer_var.trace("w",
                             lambda *args: self.update_temp_settings("use_timer", self.timer_var.get() == "with_timer"))
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["no_timer"],
                        variable=self.timer_var, value="no_timer", command=self.toggle_timer).pack(side="left", padx=8)
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["with_timer"],
                        variable=self.timer_var, value="with_timer", command=self.toggle_timer).pack(side="left",
                                                                                                     padx=8)

        self.timer_duration_frame = ttk.Frame(timer_frame)
        self.timer_var_duration.trace("w", lambda *args: self.update_temp_settings("game_time",
                                                                                   self.timer_var_duration.get()))
        ttk.Label(self.timer_duration_frame, text=LANGUAGES[self.settings.language]["game_duration"]).pack(side="left")
        self.timer_combo = ttk.Combobox(self.timer_duration_frame, textvariable=self.timer_var_duration,
                                        state="readonly", values=["1", "2", "3", "5", "10", "20"])
        self.timer_combo.pack(side="left", padx=5)
        if self.temp_settings.use_timer:
            self.timer_duration_frame.pack(side="left", padx=5)

    def setup_design_tab(self, notebook):
        design_frame = ttk.Frame(notebook, padding="10")
        notebook.add(design_frame, text=LANGUAGES[self.settings.language]["design_tab"])

        sound_container = ttk.Frame(design_frame)
        sound_container.pack(fill="x", pady=2)
        ttk.Label(sound_container, text=LANGUAGES[self.settings.language]["sound_enabled"]).pack(side="left", padx=5)
        ttk.Checkbutton(sound_container, variable=self.sound_var,
                        command=lambda: self.update_temp_settings("sound_enabled", self.sound_var.get())).pack(
            side="left", padx=5)

        player_1_color_container = ttk.Frame(design_frame)
        player_1_color_container.pack(fill="x", pady=2)
        ttk.Label(player_1_color_container, text=LANGUAGES[self.settings.language]["player_1_color"]).pack(side="left",
                                                                                                           padx=5)
        self.p1_color_button = ttk.Button(player_1_color_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.player_1_color_var, "p1_color_button"))
        self.p1_color_button.pack(side="left", padx=5)
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())

        player_2_color_container = ttk.Frame(design_frame)
        player_2_color_container.pack(fill="x", pady=2)
        ttk.Label(player_2_color_container, text=LANGUAGES[self.settings.language]["player_2_color"]).pack(side="left",
                                                                                                           padx=5)
        self.p2_color_button = ttk.Button(player_2_color_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.player_2_color_var, "p2_color_button"))
        self.p2_color_button.pack(side="left", padx=5)
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())

        board_color_1_container = ttk.Frame(design_frame)
        board_color_1_container.pack(fill="x", pady=2)
        ttk.Label(board_color_1_container, text=LANGUAGES[self.settings.language]["board_color_1"]).pack(side="left",
                                                                                                         padx=5)
        self.b1_color_button = ttk.Button(board_color_1_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.board_color_1_var, "b1_color_button"))
        self.b1_color_button.pack(side="left", padx=5)
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())

        board_color_2_container = ttk.Frame(design_frame)
        board_color_2_container.pack(fill="x", pady=2)
        ttk.Label(board_color_2_container, text=LANGUAGES[self.settings.language]["board_color_2"]).pack(side="left",
                                                                                                         padx=5)
        self.b2_color_button = ttk.Button(board_color_2_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.board_color_2_var, "b2_color_button"))
        self.b2_color_button.pack(side="left", padx=5)
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())

        piece_style_container = ttk.Frame(design_frame)
        piece_style_container.pack(fill="x", pady=2)
        ttk.Label(piece_style_container, text=LANGUAGES[self.settings.language]["piece_style"]).pack(side="left",
                                                                                                     padx=5)
        piece_styles = ["circle", "outlined_circle", "square", "diamond", "star", "custom"]
        ttk.OptionMenu(piece_style_container, self.piece_style_var, self.temp_settings.piece_style,
                       *piece_styles,
                       command=lambda _: self.update_temp_settings("piece_style", self.piece_style_var.get())).pack(
            side="left", fill="x", expand=True, padx=5)

        preview_container = ttk.Frame(design_frame)
        preview_container.pack(fill="x", pady=5)
        ttk.Label(preview_container, text="Piece Preview").pack(side="left", padx=5)
        self.preview_canvas = tk.Canvas(preview_container, width=50, height=50, bg="white")
        self.preview_canvas.pack(side="left", padx=5)
        self.update_piece_preview()

        images_container = ttk.Frame(design_frame)
        images_container.pack(fill="x", pady=5)
        ttk.Label(images_container, text=LANGUAGES[self.settings.language]["piece_images"]).pack(side="left", padx=5)

        for piece in ["player_1_piece", "player_1_king", "player_2_piece", "player_2_king"]:
            container = ttk.Frame(design_frame)
            container.pack(fill="x", pady=2)
            ttk.Label(container, text=LANGUAGES[self.settings.language][piece]).pack(side="left", padx=5)
            var = tk.StringVar(value=getattr(self.temp_settings, f"{piece}_image"))
            entry = ttk.Entry(container, textvariable=var)
            entry.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Button(container, text=LANGUAGES[self.settings.language]["upload_image"],
                       command=lambda p=piece, v=var: self.upload_image(p, v)).pack(side="left", padx=5)
            self.entries[piece] = var

    def setup_player_tab(self, notebook):
        player_frame = ttk.Frame(notebook, padding="10")
        notebook.add(player_frame, text=LANGUAGES[self.settings.language]["player_tab"])

        side: Literal["left", "right"] = "right" if self.settings.language == "fa" else "left"
        anchor = "e" if side == "left" else "w"

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_1_name"]).grid(row=0, column=0, padx=5,
                                                                                              pady=5, sticky=anchor)
        ttk.Entry(player_frame, textvariable=self.player_1_name_var, width=15).grid(row=0, column=1, padx=5, pady=5,
                                                                                    sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("player_1"), style="Custom.TButton").grid(row=0, column=2, padx=5,
                                                                                               pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_2_name"]).grid(row=1, column=0, padx=5,
                                                                                              pady=5, sticky=anchor)
        ttk.Entry(player_frame, textvariable=self.player_2_name_var, width=15).grid(row=1, column=1, padx=5, pady=5,
                                                                                    sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("player_2"), style="Custom.TButton").grid(row=1, column=2, padx=5,
                                                                                               pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["al1_name"]).grid(row=2, column=0, padx=5,
                                                                                         pady=5, sticky=anchor)
        ttk.Entry(player_frame, textvariable=self.al1_name_var, width=15).grid(row=2, column=1, padx=5, pady=5,
                                                                               sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("al1"), style="Custom.TButton").grid(row=2, column=2, padx=5,
                                                                                          pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["al2_name"]).grid(row=3, column=0, padx=5,
                                                                                         pady=5, sticky=anchor)
        ttk.Entry(player_frame, textvariable=self.al2_name_var, width=15).grid(row=3, column=1, padx=5, pady=5,
                                                                               sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("al2"), style="Custom.TButton").grid(row=3, column=2, padx=5,
                                                                                          pady=5, sticky="w")

    def setup_advanced_tab(self, notebook):
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_tab"])

        ttk.Button(advanced_frame, text=LANGUAGES[self.settings.language]["open_advanced_config"],
                   command=self.open_advanced_config_window, style="Custom.TButton").pack(pady=10)

    def open_advanced_config_window(self):
        AdvancedConfigWindow(self.window, self.settings, self.temp_settings, self.settings.language)

    def save_advanced_settings(self):
        try:
            player1_training_params = {
                "learning_rate": self.player1_learning_rate_var.get(),
                "gamma": self.player1_gamma_var.get(),
                "batch_size": self.player1_batch_size_var.get(),
                "memory_size": self.player1_memory_size_var.get()
            }
            player1_reward_weights = {
                "piece_difference": self.player1_piece_difference_var.get(),
                "king_bonus": self.player1_king_bonus_var.get(),
                "position_bonus": self.player1_position_bonus_var.get(),
                "capture_bonus": self.player1_capture_bonus_var.get(),
                "multi_jump_bonus": self.player1_multi_jump_bonus_var.get(),
                "king_capture_bonus": self.player1_king_capture_bonus_var.get(),
                "mobility_bonus": self.player1_mobility_bonus_var.get(),
                "safety_penalty": self.player1_safety_penalty_var.get()
            }
            self.temp_settings.ai_configs["player_1"]["training_params"] = player1_training_params
            self.temp_settings.ai_configs["player_1"]["reward_weights"] = player1_reward_weights

            player2_training_params = {
                "learning_rate": self.player2_learning_rate_var.get(),
                "gamma": self.player2_gamma_var.get(),  # اصلاح gamma_var به player2_gamma_var
                "batch_size": self.player2_batch_size_var.get(),
                "memory_size": self.player2_memory_size_var.get()
            }
            player2_reward_weights = {
                "piece_difference": self.player2_piece_difference_var.get(),
                "king_bonus": self.player2_king_bonus_var.get(),
                "position_bonus": self.player2_position_bonus_var.get(),
                "capture_bonus": self.player2_capture_bonus_var.get(),
                "multi_jump_bonus": self.player2_multi_jump_bonus_var.get(),
                "king_capture_bonus": self.player2_king_capture_bonus_var.get(),
                "mobility_bonus": self.player2_mobility_bonus_var.get(),
                "safety_penalty": self.player2_safety_penalty_var.get()
            }
            self.temp_settings.ai_configs["player_2"]["training_params"] = player2_training_params
            self.temp_settings.ai_configs["player_2"]["reward_weights"] = player2_reward_weights

            ai_config = load_ai_config()
            ai_config["ai_configs"]["player_1"].update(self.temp_settings.ai_configs["player_1"])
            ai_config["ai_configs"]["player_2"].update(self.temp_settings.ai_configs["player_2"])
            save_ai_config(ai_config)

            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_saved"]
            )
        except Exception as e:
            self.log_error(f"خطا در ذخیره تنظیمات پیشرفته: {str(e)}")

    def toggle_ai_vs_ai_options(self):
        if self.play_with_var.get() == "ai_vs_ai":
            self.repeat_once_rb.config(state="normal")
            self.repeat_until_rb.config(state="normal")
        else:
            self.repeat_once_rb.config(state="disabled")
            self.repeat_until_rb.config(state="disabled")
            self.repeat_hands_frame.pack_forget()

    def toggle_repeat_options(self):
        if self.ai_vs_ai_var.get() == "repeat_game" and self.play_with_var.get() == "ai_vs_ai":
            self.repeat_hands_frame.pack(side="left", padx=5)
        else:
            self.repeat_hands_frame.pack_forget()

    def toggle_timer(self):
        if self.timer_var.get() == "with_timer":
            self.timer_duration_frame.pack(side="left", padx=5)
        else:
            self.timer_duration_frame.pack_forget()

    def choose_color(self, color_var: tk.StringVar, button: str):
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            color_var.set(color)
            self.update_color_button(button, color)

    @staticmethod
    def update_color_button(button: ttk.Button, color: str):
        try:
            style_name = f"Color_{id(button)}.TButton"
            style = ttk.Style()
            style.configure(style_name, background=color)
            button.configure(style=style_name)
        except Exception as e:
            logging.error(f"Error updating button color: {str(e)}")

    def upload_image(self, player: str, var: tk.StringVar = None):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            if var:
                var.set(file_path)
            if player == "player_1":
                self.temp_settings.player_1_image = file_path
            elif player == "player_2":
                self.temp_settings.player_2_image = file_path
            elif player == "al1":
                self.temp_settings.al1_image = file_path
            elif player == "al2":
                self.temp_settings.al2_image = file_path
            elif player in ["player_1_piece", "player_1_king", "player_2_piece", "player_2_king"]:
                setattr(self.temp_settings, f"{player}_image", file_path)

    def update_piece_preview(self):
        self.preview_canvas.delete("all")
        style = self.piece_style_var.get()
        color = hex_to_rgb(self.player_1_color_var.get())
        center_x, center_y = 25, 25
        radius = 20

        if style == "circle":
            self.preview_canvas.create_oval(center_x - radius, center_y - radius,
                                            center_x + radius, center_y + radius,
                                            fill=rgb_to_hex(color))
        elif style == "outlined_circle":
            self.preview_canvas.create_oval(center_x - radius, center_y - radius,
                                            center_x + radius, center_y + radius,
                                            fill=rgb_to_hex(color), outline="black", width=2)
        elif style == "square":
            self.preview_canvas.create_rectangle(center_x - radius, center_y - radius,
                                                 center_x + radius, center_y + radius,
                                                 fill=rgb_to_hex(color))
        elif style == "diamond":
            points = [
                center_x, center_y - radius,
                          center_x + radius, center_y,
                center_x, center_y + radius,
                          center_x - radius, center_y
            ]
            self.preview_canvas.create_polygon(points, fill=rgb_to_hex(color), outline="black", width=2)
        elif style == "star":
            points = [
                center_x, center_y - radius,
                          center_x + radius * 0.3, center_y - radius * 0.3,
                          center_x + radius, center_y,
                          center_x + radius * 0.3, center_y + radius * 0.3,
                center_x, center_y + radius,
                          center_x - radius * 0.3, center_y + radius * 0.3,
                          center_x - radius, center_y,
                          center_x - radius * 0.3, center_y - radius * 0.3
            ]
            self.preview_canvas.create_polygon(points, fill=rgb_to_hex(color), outline="black", width=2)
        elif style == "custom":
            self.preview_canvas.create_text(center_x, center_y, text="Custom", fill="black")

    def save(self):
        try:
            pause_time = self.ai_pause_var.get()
            if not 0 <= pause_time <= 5000:
                self.log_error(LANGUAGES[self.settings.language]["ai_pause_error"])
                return

            repeat_hands = self.hmh_var.get()
            if self.ai_vs_ai_var.get() == "repeat_game" and not 1 <= repeat_hands <= 1000:
                self.log_error(LANGUAGES[self.settings.language]["invalid_number_hands"])
                return

            ai_config = load_ai_config()
            for ai_type, var in [
                (self.player_1_ai_type_var.get(), LANGUAGES[self.settings.language]["player_1_ai"]),
                (self.player_2_ai_type_var.get(), LANGUAGES[self.settings.language]["player_2_ai"])
            ]:
                if ai_type != "none" and ai_type not in ai_config["ai_types"]:
                    self.log_error(f"{var}: نوع '{ai_type}' پیدا نشد")
                    return

            config = load_config()
            config.update({
                "language": self.temp_settings.language,
                "game_mode": self.temp_settings.game_mode,
                "ai_vs_ai_mode": self.temp_settings.ai_vs_ai_mode,
                "repeat_hands": repeat_hands,
                "player_starts": self.temp_settings.player_starts,
                "use_timer": self.temp_settings.use_timer,
                "game_time": self.temp_settings.game_time,
                "piece_style": self.piece_style_var.get(),
                "sound_enabled": self.sound_var.get(),
                "ai_pause_time": pause_time,
                "player_1_name": self.player_1_name_var.get(),
                "player_2_name": self.player_2_name_var.get(),
                "al1_name": self.al1_name_var.get(),
                "al2_name": self.al2_name_var.get(),
                "player_1_color": self.player_1_color_var.get(),
                "player_2_color": self.player_2_color_var.get(),
                "board_color_1": self.board_color_1_var.get(),
                "board_color_2": self.board_color_2_var.get(),
                "player_1_image": self.entries.get("player_1_image",
                                                   tk.StringVar(value=self.temp_settings.player_1_image)).get(),
                "player_2_image": self.entries.get("player_2_image",
                                                   tk.StringVar(value=self.temp_settings.player_2_image)).get(),
                "al1_image": self.entries.get("al1_image", tk.StringVar(value=self.temp_settings.al1_image)).get(),
                "al2_image": self.entries.get("al2_image", tk.StringVar(value=self.temp_settings.al2_image)).get(),
                "player_1_piece_image": self.entries.get("player_1_piece", tk.StringVar(
                    value=self.temp_settings.player_1_piece_image)).get(),
                "player_1_king_image": self.entries.get("player_1_king", tk.StringVar(
                    value=self.temp_settings.player_1_king_image)).get(),
                "player_2_piece_image": self.entries.get("player_2_piece", tk.StringVar(
                    value=self.temp_settings.player_2_piece_image)).get(),
                "player_2_king_image": self.entries.get("player_2_king", tk.StringVar(
                    value=self.temp_settings.player_2_king_image)).get(),
                "player_1_ai_type": self.player_1_ai_type_var.get(),
                "player_2_ai_type": self.player_2_ai_type_var.get()
            })
            save_config(config)

            for player in ["player_1", "player_2"]:
                ai_type = self.temp_settings.ai_configs[player]["ai_type"]
                ai_config["ai_configs"][player] = {
                    "ai_type": ai_type,
                    "ai_code": self.temp_settings.ai_configs[player]["ai_code"],
                    "ability_level": self.temp_settings.ai_configs[player]["ability_level"],
                    "params": self.temp_settings.ai_configs[player]["params"]
                }
                if ai_type != "none":
                    code = ai_config["ai_types"][ai_type]["code"]
                    ai_specific_config = load_ai_specific_config(code)
                    ai_specific_config[player] = self.temp_settings.ai_configs[player]["params"]
                    save_ai_specific_config(code, ai_specific_config)

            save_ai_config(ai_config)

            if config.get("language") != self.interface.settings.language:
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["warning"],
                    LANGUAGES[self.settings.language]["restart_required"],
                    parent=self.window
                )

            self.interface.pending_settings = config
            if not self.interface.game.game_started or self.interface.game.game_over:
                self.interface.apply_pending_settings()
            else:
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["warning"],
                    LANGUAGES[self.settings.language]["apply_after_game"],
                    parent=self.window
                )

            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_saved"],
                parent=self.window
            )
            self.close()

        except Exception as e:
            self.log_error(f"خطا در ذخیره تنظیمات: {str(e)}")

    def reset(self):
        print("Reset called")
        if not messagebox.askyesno(
                LANGUAGES[self.settings.language]["warning"],
                LANGUAGES[self.settings.language]["confirm_reset_all"],
                parent=self.window
        ):
            print("Reset cancelled by user")
            return

        try:
            ai_config = load_ai_config()
            ai_types = list(ai_config["ai_types"].keys())
            default_ai_type = ai_types[0] if ai_types else "none"
            current_player_1_ai_type = self.temp_settings.player_1_ai_type if self.temp_settings.player_1_ai_type in ai_types else "none"
            current_player_2_ai_type = self.temp_settings.player_2_ai_type if self.temp_settings.player_2_ai_type in ai_types else default_ai_type

            self.temp_settings = type(self.temp_settings)()
            self.temp_settings.language = "en"
            self.temp_settings.game_mode = "human_vs_ai"
            self.temp_settings.ai_vs_ai_mode = "only_once"
            self.temp_settings.repeat_hands = 10
            self.temp_settings.player_starts = True
            self.temp_settings.use_timer = True
            self.temp_settings.game_time = 5
            self.temp_settings.piece_style = "circle"
            self.temp_settings.sound_enabled = False
            self.temp_settings.ai_pause_time = 20
            self.temp_settings.player_1_name = "Player 1"
            self.temp_settings.player_2_name = "AI Player"
            self.temp_settings.al1_name = "AI 1"
            self.temp_settings.al2_name = "AI 2"
            self.temp_settings.player_1_color = (255, 0, 0)
            self.temp_settings.player_2_color = (0, 0, 255)
            self.temp_settings.board_color_1 = (255, 255, 255)
            self.temp_settings.board_color_2 = (139, 69, 19)
            self.temp_settings.player_1_ai_type = current_player_1_ai_type
            self.temp_settings.player_2_ai_type = current_player_2_ai_type
            self.temp_settings.player_1_image = ""
            self.temp_settings.player_2_image = ""
            self.temp_settings.al1_image = ""
            self.temp_settings.al2_image = ""
            self.temp_settings.player_1_piece_image = str(Path(self.config["assets_dir"]) / "pieces" / "red_piece.png")
            self.temp_settings.player_1_king_image = str(Path(self.config["assets_dir"]) / "pieces" / "red_king.png")
            self.temp_settings.player_2_piece_image = str(Path(self.config["assets_dir"]) / "pieces" / "blue_piece.png")
            self.temp_settings.player_2_king_image = str(Path(self.config["assets_dir"]) / "pieces" / "blue_king.png")
            self.temp_settings.ai_configs = {
                "player_1": {
                    "ai_type": current_player_1_ai_type,
                    "ai_code": ai_config["ai_types"].get(current_player_1_ai_type, {}).get("code", None),
                    "ability_level": 5,
                    "params": DEFAULT_AI_PARAMS.copy() if current_player_1_ai_type != "none" else {}
                },
                "player_2": {
                    "ai_type": current_player_2_ai_type,
                    "ai_code": ai_config["ai_types"].get(current_player_2_ai_type, {}).get("code", None),
                    "ability_level": 5,
                    "params": DEFAULT_AI_PARAMS.copy() if current_player_2_ai_type != "none" else {}
                }
            }

            ui_vars = {
                "lang_var": self.temp_settings.language,
                "play_with_var": self.temp_settings.game_mode,
                "ai_vs_ai_var": self.temp_settings.ai_vs_ai_mode,
                "hmh_var": self.temp_settings.repeat_hands,
                "start_var": "player_1" if self.temp_settings.player_starts else "player_2",
                "timer_var": "with_timer" if self.temp_settings.use_timer else "no_timer",
                "timer_var_duration": self.temp_settings.game_time,
                "piece_style_var": self.temp_settings.piece_style,
                "sound_var": self.temp_settings.sound_enabled,
                "ai_pause_var": self.temp_settings.ai_pause_time,
                "player_1_name_var": self.temp_settings.player_1_name,
                "player_2_name_var": self.temp_settings.player_2_name,
                "al1_name_var": self.temp_settings.al1_name,
                "al2_name_var": self.temp_settings.al2_name,
                "player_1_color_var": rgb_to_hex(self.temp_settings.player_1_color),
                "player_2_color_var": rgb_to_hex(self.temp_settings.player_2_color),
                "board_color_1_var": rgb_to_hex(self.temp_settings.board_color_1),
                "board_color_2_var": rgb_to_hex(self.temp_settings.board_color_2),
                "player_1_ai_type_var": self.temp_settings.player_1_ai_type,
                "player_2_ai_type_var": self.temp_settings.player_2_ai_type
            }
            for var_name, value in ui_vars.items():
                if hasattr(self, var_name):
                    getattr(self, var_name).set(value)
                else:
                    print(f"Warning: {var_name} not found in SettingsWindow")

            for key in ["player_1_image", "player_2_image", "al1_image", "al2_image",
                        "player_1_piece", "player_1_king", "player_2_piece", "player_2_king"]:
                if key in self.entries:
                    self.entries[key].set(self.temp_settings.__dict__.get(key, ""))
                else:
                    print(f"Warning: Entry {key} not found in self.entries")

            self.update_ui()
            print("Temp settings after reset:", vars(self.temp_settings))
            self.save()
            print("Reset completed, settings saved")
            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_reset"]
            )
        except Exception as e:
            print(f"Error in reset: {str(e)}")
            self.log_error(f"خطا در بازنشانی تنظیمات: {str(e)}")

    def update_ui(self):
        if not self.window or not self.is_open:
            return
        self.lang_var.set(self.temp_settings.language)
        self.play_with_var.set(self.temp_settings.game_mode)
        self.ai_vs_ai_var.set(self.temp_settings.ai_vs_ai_mode)
        self.hmh_var.set(self.temp_settings.repeat_hands)
        self.start_var.set("player_1" if self.temp_settings.player_starts else "player_2")
        self.timer_var.set("with_timer" if self.temp_settings.use_timer else "no_timer")
        self.timer_var_duration.set(self.temp_settings.game_time)
        self.piece_style_var.set(self.temp_settings.piece_style)
        self.sound_var.set(self.temp_settings.sound_enabled)
        self.ai_pause_var.set(self.temp_settings.ai_pause_time)
        self.player_1_name_var.set(self.temp_settings.player_1_name)
        self.player_2_name_var.set(self.temp_settings.player_2_name)
        self.al1_name_var.set(self.temp_settings.al1_name)
        self.al2_name_var.set(self.temp_settings.al2_name)
        self.player_1_color_var.set(rgb_to_hex(self.temp_settings.player_1_color))
        self.player_2_color_var.set(rgb_to_hex(self.temp_settings.player_2_color))
        self.board_color_1_var.set(rgb_to_hex(self.temp_settings.board_color_1))
        self.board_color_2_var.set(rgb_to_hex(self.temp_settings.board_color_2))
        self.player_1_ai_type_var.set(self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"))
        self.player_2_ai_type_var.set(self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none"))
        ability_mapping = {
            1: "very_weak",
            3: "weak",
            5: "medium",
            7: "strong",
            9: "very_strong"
        }
        self.player_1_ability_var.set(
            LANGUAGES[self.settings.language][
                ability_mapping.get(self.temp_settings.ai_configs["player_1"]["ability_level"], "medium")
            ]
        )
        self.player_2_ability_var.set(
            LANGUAGES[self.settings.language][
                ability_mapping.get(self.temp_settings.ai_configs["player_2"]["ability_level"], "medium")
            ]
        )
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())
        self.toggle_ai_vs_ai_options()
        self.toggle_repeat_options()
        self.toggle_timer()
        self.update_ai_dropdowns()


class AIProgressWindow(BaseWindow):
    def __init__(self, interface, root=None):
        super().__init__(interface, root)

    def create_widgets(self):
        self.create_window(
            LANGUAGES[self.settings.language]["ai_progress"],
            self.config.get("progress_window_width", 800),
            self.config.get("progress_window_height", 600)
        )

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        model_files = list(self.pth_dir.glob("*.pth"))
        if not model_files:
            ttk.Label(self.window, text=LANGUAGES[self.settings.language]["model_not_found"]).pack(pady=20)
            ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                       command=self.close, style="Custom.TButton").pack(pady=10)
            return

        for pth_file in model_files:
            model_id = pth_file.stem
            self.setup_model_tab(notebook, pth_file, model_id)

        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(pady=10)

    def setup_model_tab(self, notebook, pth_file: Path, model_id: str):
        tab_frame = ttk.Frame(notebook)
        notebook.add(tab_frame, text=model_id)

        content_frame = ttk.Frame(tab_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        model_frame = ttk.LabelFrame(content_frame, text=LANGUAGES[self.settings.language]["model_parameters"],
                                     padding=10)
        model_frame.pack(fill="x", pady=5)

        headers = [
            LANGUAGES[self.settings.language]["parameter"],
            LANGUAGES[self.settings.language]["shape"],
            LANGUAGES[self.settings.language]["num_elements"]
        ]

        table_frame = ttk.Frame(model_frame)
        table_frame.pack(fill="both", expand=True)

        for column, head in enumerate(headers):  # تغییر col به column و header به head
            ttk.Label(table_frame, text=head, font=("Arial", 10, "bold")).grid(row=0, column=column, padx=5, pady=2,
                                                                               sticky="w")

        model_data = {}
        total_params = 0
        try:
            model_data = torch.load(pth_file, map_location=torch.device("cpu"))
            for key, tensor in model_data.items():
                total_params += tensor.numel()
        except Exception as e:
            model_data = {"Error": LANGUAGES[self.settings.language]["model_load_error"].format(error=str(e))}
            logging.error(f"Failed to load model {pth_file}: {str(e)}")

        if "Error" not in model_data:
            for row_idx, (key, tensor) in enumerate(model_data.items(), 1):  # تغییر row به row_idx
                ttk.Label(table_frame, text=key).grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(table_frame, text=str(list(tensor.shape))).grid(row=row_idx, column=1, sticky="w", padx=5,
                                                                          pady=2)
                ttk.Label(table_frame, text=str(tensor.numel())).grid(row=row_idx, column=2, sticky="w", padx=5, pady=2)

            ttk.Label(table_frame, text=LANGUAGES[self.settings.language]["total_parameters"],
                      font=("Arial", 10, "bold")).grid(row=row_idx + 1, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(table_frame, text=str(total_params)).grid(row=row_idx + 1, column=2, sticky="w", padx=5, pady=2)
        else:
            ttk.Label(table_frame, text=model_data["Error"]).grid(row=1, column=0, columnspan=3, padx=5, pady=10)

        progress_frame = ttk.LabelFrame(content_frame, text=LANGUAGES[self.settings.language]["training_progress"],
                                        padding=10)
        progress_frame.pack(fill="x", pady=5)

        progress_file_path = self.pth_dir / f"progress_tracker_{model_id}.json"  # تغییر progress_file به progress_file_path
        progress_data = self.load_progress_data(progress_file_path)

        if "Error" not in progress_data:
            progress_table = ttk.Frame(progress_frame)
            progress_table.pack(fill="both", expand=True)

            progress_headers = ["Epoch", "Loss", "Accuracy", "Training Time (s)"]
            if self.settings.language == "fa":
                progress_headers = ["دوره", "خطا", "دقت", "زمان آموزش (ثانیه)"]

            for column, head in enumerate(progress_headers):  # تغییر col به column و header به head
                ttk.Label(progress_table, text=head, font=("Arial", 10, "bold")).grid(row=0, column=column, padx=5,
                                                                                      pady=2, sticky="w")

            for row_idx, epoch_data in enumerate(progress_data.get("epochs", []), 1):  # تغییر row به row_idx
                ttk.Label(progress_table, text=str(epoch_data.get("epoch", "-"))).grid(row=row_idx, column=0,
                                                                                       sticky="w",
                                                                                       padx=5, pady=2)
                ttk.Label(progress_table, text=str(epoch_data.get("loss", "-"))).grid(row=row_idx, column=1, sticky="w",
                                                                                      padx=5, pady=2)
                ttk.Label(progress_table, text=str(epoch_data.get("accuracy", "-"))).grid(row=row_idx, column=2,
                                                                                          sticky="w",
                                                                                          padx=5, pady=2)
                ttk.Label(progress_table, text=str(epoch_data.get("training_time", "-"))).grid(row=row_idx, column=3,
                                                                                               sticky="w", padx=5,
                                                                                               pady=2)
        else:
            ttk.Label(progress_frame, text=progress_data["Error"]).grid(row=1, column=0, padx=5, pady=10)

    def load_progress_data(self, progress_file_path: Path) -> dict:  # تغییر progress_file به progress_file_path
        try:
            if progress_file_path.exists():
                with open(progress_file_path, "r") as f:
                    data = json.load(f)
                return data
            else:
                return {"Error": LANGUAGES[self.settings.language]["model_not_found"]}
        except Exception as e:
            logging.error(f"Failed to load progress file {progress_file_path}: {str(e)}")
            return {"Error": LANGUAGES[self.settings.language]["model_load_error"].format(error=str(e))}


class HelpWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["help"],
                           self.config.get("help_window_width", 300),
                           self.config.get("help_window_height", 200))
        ttk.Label(self.window, text="COMING SOON!", font=("Arial", 14)).pack(pady=50)
        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(pady=10)


class AboutWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["about_me"],
                           self.config.get("about_window_width", 300),
                           self.config.get("about_window_height", 200))
        ttk.Label(self.window, text="COMING SOON!", font=("Arial", 14)).pack(pady=50)
        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(pady=10)


class AdvancedConfigWindow(tk.Toplevel):
    def __init__(self, parent, settings, temp_settings, language):
        super().__init__(parent)
        self.parent = parent
        self.settings = settings
        self.temp_settings = temp_settings
        self.language = language
        self.title(LANGUAGES[language]["advanced_settings_title"])

        # بارگذاری تنظیمات پنجره
        config = load_config()  # فرض می‌کنیم load_config تعریف شده است
        width = config.get("settings_window_width", 500)
        height = config.get("settings_window_height", 750)
        x = (config.get("window_width", 800) - width) // 2
        y = (config.get("window_height", 600) - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.resizable(True, True)
        self.minsize(config.get("min_window_width", 300), config.get("min_window_height", 200))

        # بارگذاری تنظیمات AI
        self.ai_config = load_ai_config()
        self.ai_types = list(self.ai_config.get("ai_types", {}).keys())
        self.param_vars = {}
        self.current_ai_type = self.ai_types[0] if self.ai_types else None
        if not self.ai_types:
            messagebox.showwarning(
                LANGUAGES[self.language]["warning"],
                "No AI types available. Please add AI modules.",
                parent=self
            )

        # تنظیمات رابط کاربری
        main_frame = ttk.Frame(self)
        main_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main_frame)
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        notebook = ttk.Notebook(self.scrollable_frame)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        for ai_type in self.ai_types:
            self.setup_ai_tab(notebook, ai_type)

        notebook.bind("<<NotebookTabChanged>>", self.update_current_ai)

        button_frame = ttk.Frame(self)
        button_frame.pack(side="bottom", fill="x", pady=10)

        ttk.Button(button_frame, text=LANGUAGES[language]["save_all"],
                   command=self.save_advanced_settings,
                   style="Custom.TButton").pack(side="right", padx=5)
        self.reset_buttons = {}
        for player in ["player_1", "player_2"]:
            self.reset_buttons[player] = ttk.Button(
                button_frame,
                text=f"{LANGUAGES[language]['reset']} {LANGUAGES[language][player]}",
                command=lambda p=player: self.reset_player_tab(self.current_ai_type, p),
                style="Custom.TButton"
            )
            self.reset_buttons[player].pack(side="right", padx=5)

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def setup_player_subtab(self, sub_notebook, ai_type: str, player: str):
        frame = ttk.Frame(sub_notebook, padding="10")
        sub_notebook.add(frame, text=LANGUAGES[self.language][player])

        ai_code = self.ai_config["ai_types"][ai_type]["code"]
        ai_specific_config = load_ai_specific_config(ai_code)
        player_params = ai_specific_config.get(player, DEFAULT_AI_PARAMS.copy())

        if ai_type not in self.param_vars:
            self.param_vars[ai_type] = {}
        self.param_vars[ai_type][player] = {}

        for param_category, params in DEFAULT_AI_PARAMS.items():
            category_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.language].get(param_category, param_category),
                                            padding=10)
            category_frame.pack(fill="x", pady=5)

            # اعتبارسنجی اینکه params یک دیکشنری است
            if not isinstance(params, dict):
                logging.error(f"Invalid params structure for category {param_category}: {params}")
                continue

            row_idx = 0
            for param_name, default_value in params.items():
                if param_name == "fc_layer_sizes":
                    continue
                var_type = tk.StringVar if param_name == "cache_file" else tk.DoubleVar if isinstance(default_value, float) else tk.IntVar
                value = player_params.get(param_category, {}).get(param_name, default_value)
                self.param_vars[ai_type][player][param_name] = var_type(value=value)

                ttk.Label(category_frame, text=param_name).grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
                ttk.Entry(category_frame, textvariable=self.param_vars[ai_type][player][param_name]).grid(
                    row=row_idx, column=1, sticky="w", padx=5, pady=2
                )
                row_idx += 1

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_current_ai(self, event):
        notebook = event.widget
        current_tab = notebook.select()
        tab_index = notebook.index(current_tab)
        self.current_ai_type = self.ai_types[tab_index] if self.ai_types else None
        logging.debug(f"Current AI type updated to: {self.current_ai_type}")

    def setup_ai_tab(self, notebook, ai_type: str):
        ai_frame = ttk.Frame(notebook, padding="10")
        description = self.ai_config["ai_types"][ai_type].get("description", ai_type)
        notebook.add(ai_frame, text=f"{ai_type}")

        sub_notebook = ttk.Notebook(ai_frame)
        sub_notebook.pack(fill="both", expand=True)

        for player in ["player_1", "player_2"]:
            self.setup_player_subtab(sub_notebook, ai_type, player)

    def reset_player_tab(self, ai_type, player):
        if not ai_type:
            logging.error("No AI type selected, aborting reset")
            messagebox.showerror(
                LANGUAGES[self.language]["error"],
                LANGUAGES[self.language]["no_ai_selected"],
                parent=self
            )
            return

        if not messagebox.askyesno(
                LANGUAGES[self.language]["warning"],
                f"{LANGUAGES[self.language]['confirm_reset']} {LANGUAGES[self.language][player]}",
                parent=self
        ):
            logging.info(f"Reset for {player} cancelled by user")
            return

        player_params = DEFAULT_AI_PARAMS.copy()
        ai_code = self.ai_config["ai_types"][ai_type]["code"]
        ai_specific_config = load_ai_specific_config(ai_code)

        for param_category, params in DEFAULT_AI_PARAMS.items():
            if not isinstance(params, dict):
                logging.error(f"Invalid params structure for category {param_category}: {params}")
                continue
            for param_name, default_value in params.items():
                if param_name == "fc_layer_sizes":
                    player_params[param_category][param_name] = default_value
                    continue
                param_var = self.param_vars[ai_type][player].get(param_name)
                if param_var:
                    try:
                        if param_name == "cache_file":
                            value = str(default_value)
                        elif param_name == "cache_save_interval":
                            value = int(default_value)
                        elif isinstance(default_value, float):
                            value = float(default_value)
                        else:
                            value = int(default_value)
                        param_var.set(value)
                        player_params[param_category][param_name] = value
                    except Exception as e:
                        logging.error(f"Error setting {param_category}.{param_name}: {e}")
                        param_var.set(default_value)
                        player_params[param_category][param_name] = default_value
                    logging.debug(f"Set {param_category}.{param_name} to {param_var.get()}")

        self.temp_settings.ai_configs[player]["params"] = player_params
        ai_specific_config[player] = player_params
        save_ai_specific_config(ai_code, ai_specific_config)
        logging.info(f"Saved reset params for {player} to {ai_code}_config.json")

        messagebox.showinfo(
            LANGUAGES[self.language]["info"],
            f"{LANGUAGES[self.language]['settings_reset']} {LANGUAGES[self.language][player]}",
            parent=self
        )

    def save_advanced_settings(self):
        try:
            for ai_type in self.ai_types:
                ai_code = self.ai_config["ai_types"][ai_type]["code"]
                ai_specific_config = load_ai_specific_config(ai_code)

                for player in ["player_1", "player_2"]:
                    player_params = {}
                    for param_category, params in DEFAULT_AI_PARAMS.items():
                        player_params[param_category] = {}
                        if not isinstance(params, dict):
                            logging.error(f"Invalid params structure for category {param_category}: {params}")
                            continue
                        for param_name, default_value in params.items():
                            param_var = self.param_vars[ai_type][player].get(param_name)
                            if param_var:
                                try:
                                    if param_name == "cache_file":
                                        value = param_var.get()
                                    elif param_name == "cache_save_interval":
                                        value = int(param_var.get())
                                    elif isinstance(default_value, float):
                                        value = float(param_var.get())
                                    else:
                                        value = int(param_var.get())
                                    player_params[param_category][param_name] = value
                                except (ValueError, TypeError) as e:
                                    player_params[param_category][param_name] = default_value
                                    logging.error(
                                        f"Invalid value for {param_category}.{param_name} in {ai_type} for {player}: {e}")
                            else:
                                player_params[param_category][param_name] = default_value

                    self.temp_settings.ai_configs[player]["params"] = player_params
                    ai_specific_config[player] = player_params
                    save_ai_specific_config(ai_code, ai_specific_config)

            ai_config = load_ai_config()
            ai_config["ai_configs"]["player_1"].update(self.temp_settings.ai_configs["player_1"])
            ai_config["ai_configs"]["player_2"].update(self.temp_settings.ai_configs["player_2"])
            save_ai_config(ai_config)

            messagebox.showinfo(
                LANGUAGES[self.language]["info"],
                LANGUAGES[self.language]["settings_saved"],
                parent=self
            )
            self.destroy()

        except Exception as e:
            logging.error(f"Error saving advanced settings: {e}")
            messagebox.showerror(
                LANGUAGES[self.language]["error"],
                LANGUAGES[self.language]["error_saving_settings"].format(error=str(e)),
                parent=self
            )





