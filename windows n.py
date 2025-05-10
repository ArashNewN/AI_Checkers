import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from .game import Game
from .settings import GameSettings
from .config import save_config, load_config, load_ai_config
from .constants import LANGUAGES
from .utils import hex_to_rgb, rgb_to_hex
import os
import json
import sys
import importlib
from pathlib import Path


# تعریف استایل کلی برای برنامه
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
        self.is_open = False
        self.config = load_config()
        configure_styles()

    def create_window(self, title, width=None, height=None):
        if self.is_open or self.window:
            return
        self.window = tk.Toplevel(self.root) if self.root else tk.Toplevel()
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

class SettingsWindow(BaseWindow):
    def __init__(self, interface, root=None):
        super().__init__(interface, root)
        self.temp_settings = GameSettings()
        self.copy_current_settings()
        self.ai_types = ["none"]
        self.load_ai_config()
        self.entries = {}  # For advanced config entries
        self.player_1_training_params = tk.StringVar(value=json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("training_params", {})))
        self.player_1_reward_weights = tk.StringVar(value=json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("reward_weights", {})))
        self.player_2_training_params = tk.StringVar(value=json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("training_params", {})))
        self.player_2_reward_weights = tk.StringVar(value=json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("reward_weights", {})))

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

        side = "right" if self.settings.language == "fa" else "left"
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["save_changes"],
                   command=self.save, style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["reset_settings"],
                   command=self.reset, style="Custom.TButton").pack(side=side, padx=5)

        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def load_ai_config():
        """بارگذاری تنظیمات AI از فایل ai_config.json"""
        ai_config_path = get_ai_config_path()
        default_ai_config = {
            "ai_types": {}  # بدون AI‌های پیش‌فرض
        }
        try:
            if ai_config_path.exists():
                with open(ai_config_path, "r", encoding="utf-8") as f:
                    ai_config = json.load(f)
                    default_ai_config["ai_types"] = ai_config.get("ai_types", {})
                    print(f"Loaded AI config from {ai_config_path}")
            else:
                print(f"AI config file not found at {ai_config_path}, creating with default config")
                save_ai_config(default_ai_config)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading AI config from {ai_config_path}: {e}, using default AI config")
            save_ai_config(default_ai_config)
        # اعتبارسنجی AI‌ها (فقط برای نمایش در رابط کاربری)
        valid_ai_types = {}
        for ai_type, ai_info in default_ai_config["ai_types"].items():
            module_name = ai_info.get("module", "")
            # بررسی وجود فایل ماژول
            if os.path.exists(f"a/{module_name}.py"):
                valid_ai_types[ai_type] = ai_info
            else:
                print(f"AI type {ai_type} kept in config but not displayed: module {module_name}.py not found")
                valid_ai_types[ai_type] = ai_info  # AI در فایل کانفیگ نگه داشته می‌شود
        default_ai_config["ai_types"] = valid_ai_types
        return default_ai_config

    def check_ai_module(self, ai_type):
        if ai_type == "none":
            return True
        config = load_ai_config()
        ai_types = config.get("ai_types", {})
        if ai_type not in ai_types:
            return False
        module_name = ai_types[ai_type].get("module", "")
        if module_name.startswith("a."):
            module_name = module_name[2:]  # حذف پیشوند "a."
        return os.path.exists(f"a/{module_name}.py")

    def add_ai_type(self, ai_type, module_name, class_name, description):
        """Add a new AI type to the config."""
        if not all([ai_type, module_name, class_name]):
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["fill_all_fields"]
            )
            return
        if ai_type in self.ai_types:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["ai_type_exists"]
            )
            return
        if not self.check_ai_module(module_name):
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"ماژول {module_name}.py پیدا نشد یا قابل بارگذاری نیست"
            )
            return
        # بررسی وجود کلاس در ماژول
        try:
            module = importlib.import_module(module_name)
            if not hasattr(module, class_name):
                messagebox.showerror(
                    LANGUAGES[self.settings.language]["error"],
                    f"کلاس {class_name} در ماژول {module_name} پیدا نشد"
                )
                return
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در بررسی ماژول: {str(e)}"
            )
            return

        config = load_ai_config()
        config["ai_types"][ai_type] = {
            "module": module_name,
            "class": class_name,
            "description": description,
            "training_params": {
                "learning_rate": 0.0005,
                "gamma": 0.99,
                "batch_size": 128,
                "memory_size": 10000,
                "update_target_every": 100,
                "gradient_clip": 1.0,
                "target_update_alpha": 0.01,
                "reward~~reward_threshold": 0.5
            },
            "reward_weights": {
                "win": 1.0,
                "loss": -1.0,
                "move": 0.01,
                "king": 0.5
            }
        }
        save_config(config)
        self.ai_types.append(ai_type)
        self.player_1_ai_menu["menu"].add_command(label=ai_type, command=lambda: self.player_1_ai_type_var.set(ai_type))
        self.player_2_ai_menu["menu"].add_command(label=ai_type, command=lambda: self.player_2_ai_type_var.set(ai_type))
        self.remove_ai_menu["menu"].delete(0, "end")
        for ai in self.ai_types[1:]:
            self.remove_ai_menu["menu"].add_command(label=ai, command=lambda x=ai: self.remove_ai_var.set(x))
        messagebox.showinfo(
            LANGUAGES[self.settings.language]["info"],
            LANGUAGES[self.settings.language]["ai_added"]
        )

    def remove_ai_type(self, ai_type):
        """Remove an AI from the configuration."""
        if ai_type == "none":
            messagebox.showerror(
                LANGUAGES[self.interface.settings.language]["error"],
                "Cannot remove 'none' type"
            )
            return
        try:
            with open("ai_config.json", "r") as f:
                config = json.load(f)
            config["available_ais"] = [ai for ai in config.get("available_ais", []) if ai["type"] != ai_type]
            with open("ai_config.json", "w") as f:
                json.dump(config, f, indent=2)
            self.ai_types.remove(ai_type)
            self.update_ai_dropdowns()
            messagebox.showinfo(
                LANGUAGES[self.interface.settings.language]["success"],
                f"AI type {ai_type} removed successfully"
            )
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.interface.settings.language]["error"],
                f"Failed to remove AI type {ai_type}: {e}"
            )

    def add_new_ai(self):
        ai_type = self.new_ai_type.get().strip()
        description = self.new_description.get().strip()
        module_class = self.ai_module_class_var.get().strip()

        if not ai_type or not module_class:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["ai_type_module_required"]
            )
            return

        if ":" not in module_class:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["module_class_format"]
            )
            return

        module_name, class_name = module_class.split(":")
        try:
            module = importlib.import_module(f"a.{module_name}")
            if not hasattr(module, class_name):
                messagebox.showerror(
                    LANGUAGES[self.settings.language]["error"],
                    f"کلاس {class_name} در ماژول {module_name} پیدا نشد"
                )
                return
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در بارگذاری ماژول {module_name}: {str(e)}"
            )
            return

        config = load_ai_config()
        config["ai_types"] = config.get("ai_types", {})
        if ai_type in config["ai_types"]:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"نوع AI '{ai_type}' قبلاً وجود دارد"
            )
            return

        config["ai_types"][ai_type] = {
            "module": module_name,
            "class": class_name,
            "description": description,
            "training_params": load_config().get("default_ai_params", {}).get("training_params", {}),
            "reward_weights": load_config().get("default_ai_params", {}).get("reward_weights", {})
        }
        save_ai_config(config)
        self.newly_added_ai = ai_type
        self.ai_types = ["none"] + [t for t in config["ai_types"].keys()]
        self.update_ai_list()
        messagebox.showinfo(
            LANGUAGES[self.settings.language]["info"],
            f"AI '{ai_type}' با موفقیت اضافه شد"
        )

    def copy_current_settings(self):
        """Copy current settings to temp_settings."""
        loaded_config = load_config()
        for key, value in loaded_config.items():
            if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                value = hex_to_rgb(value)
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
            if hasattr(self.temp_settings, key):
                setattr(self.temp_settings, key, value)

    def setup_game_tab(self, notebook):
        game_frame = ttk.Frame(notebook, padding="10")
        notebook.add(game_frame, text=LANGUAGES[self.settings.language]["game_settings_tab"])

        # تنظیمات زبان
        lang_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["language"], padding=10)
        lang_frame.pack(fill="x", pady=5)
        self.lang_var = tk.StringVar(value=self.temp_settings.language)
        self.lang_var.trace("w", lambda *args: self.update_temp_settings("language", self.lang_var.get()))
        lang_menu = ttk.OptionMenu(lang_frame, self.lang_var, self.temp_settings.language, "en", "fa")
        lang_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        lang_frame.grid_columnconfigure(0, weight=1)
        lang_menu["menu"].config(font=("Arial", 10))

        # تنظیمات حالت بازی
        play_with_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["play_with"], padding=10)
        play_with_frame.pack(fill="x", pady=5)
        self.play_with_var = tk.StringVar(value=self.temp_settings.game_mode)
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

        # تنظیمات حالت AI vs AI
        self.ai_vs_ai_subframe = ttk.Frame(game_frame, padding=10)
        self.ai_vs_ai_subframe.pack(fill="x", pady=5)
        self.ai_vs_ai_var = tk.StringVar(value=self.temp_settings.ai_vs_ai_mode)
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

        # تنظیم حالت اولیه گزینه‌های تکرار
        if self.play_with_var.get() != "ai_vs_ai":
            self.repeat_once_rb.config(state="disabled")
            self.repeat_until_rb.config(state="disabled")
        else:
            self.repeat_once_rb.config(state="normal")
            self.repeat_until_rb.config(state="normal")

        self.repeat_hands_frame = ttk.Frame(self.ai_vs_ai_subframe)
        self.hmh_var = tk.IntVar(value=self.temp_settings.repeat_hands)
        self.hmh_var.trace("w", lambda *args: self.update_temp_settings("repeat_hands", self.hmh_var.get()))
        self.hmh_entry = ttk.Entry(self.repeat_hands_frame, textvariable=self.hmh_var, width=5)
        self.hmh_entry.pack(side="left", padx=5)
        self.toggle_repeat_options()

        # تنظیمات شروع‌کننده بازی
        start_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["starting_player"], padding=10)
        start_frame.pack(fill="x", pady=5)
        self.start_var = tk.StringVar(value="player_1" if self.temp_settings.player_starts else "player_2")
        self.start_var.trace("w", lambda *args: self.update_temp_settings("player_starts",
                                                                          self.start_var.get() == "player_1"))
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["player"],
                        variable=self.start_var, value="player_1").pack(side="left", padx=10)
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["ai"],
                        variable=self.start_var, value="player_2").pack(side="left", padx=10)

        # تنظیمات تایمر
        timer_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["game_timer"], padding=10)
        timer_frame.pack(fill="x", pady=5)
        self.timer_var = tk.StringVar(value="with_timer" if self.temp_settings.use_timer else "no_timer")
        self.timer_var.trace("w",
                             lambda *args: self.update_temp_settings("use_timer", self.timer_var.get() == "with_timer"))
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["no_timer"],
                        variable=self.timer_var, value="no_timer", command=self.toggle_timer).pack(side="left", padx=8)
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["with_timer"],
                        variable=self.timer_var, value="with_timer", command=self.toggle_timer).pack(side="left",
                                                                                                     padx=8)

        self.timer_duration_frame = ttk.Frame(timer_frame)
        self.timer_var_duration = tk.IntVar(value=self.temp_settings.game_time)
        self.timer_var_duration.trace("w", lambda *args: self.update_temp_settings("game_time",
                                                                                   self.timer_var_duration.get()))
        ttk.Label(self.timer_duration_frame, text=LANGUAGES[self.settings.language]["game_duration"]).pack(side="left")
        self.timer_combo = ttk.Combobox(self.timer_duration_frame, textvariable=self.timer_var_duration,
                                        state="readonly", values=[1, 2, 3, 5, 10, 20])
        self.timer_combo.pack(side="left", padx=5)
        if self.temp_settings.use_timer:
            self.timer_duration_frame.pack(side="left", padx=5)

    def update_temp_settings(self, key, value):
        """Update temp_settings with the new value for the given key."""
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
            # به‌روزرسانی تنظیمات AI
            config = load_config()
            self.temp_settings.ai_configs["player_1"] = {
                "ai_type": value,
                "ability_level": self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5),
                "training_params": config.get("ai_types", {}).get(value, {}).get("training_params",
                                                                                 config.get("default_ai_params",
                                                                                            {}).get("training_params",
                                                                                                    {})),
                "reward_weights": config.get("ai_types", {}).get(value, {}).get("reward_weights",
                                                                                config.get("default_ai_params", {}).get(
                                                                                    "reward_weights", {}))
            }
        elif key == "player_2_ai_type":
            self.temp_settings.player_2_ai_type = value
            config = load_config()
            self.temp_settings.ai_configs["player_2"] = {
                "ai_type": value,
                "ability_level": self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5),
                "training_params": config.get("ai_types", {}).get(value, {}).get("training_params",
                                                                                 config.get("default_ai_params",
                                                                                            {}).get("training_params",
                                                                                                    {})),
                "reward_weights": config.get("ai_types", {}).get(value, {}).get("reward_weights",
                                                                                config.get("default_ai_params", {}).get(
                                                                                    "reward_weights", {}))
            }
        elif key == "ai_pause_time":
            self.temp_settings.ai_pause_time = value

    def setup_design_tab(self, notebook):
        design_frame = ttk.Frame(notebook, padding="10")
        notebook.add(design_frame, text=LANGUAGES[self.settings.language]["design_tab"])

        side = "right" if self.settings.language == "fa" else "left"
        anchor = "e" if side == "left" else "w"

        style_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["piece_style"], padding=10)
        style_frame.pack(fill="x", pady=5)
        self.piece_style_var = tk.StringVar(value=self.temp_settings.piece_style)
        self.piece_style_var.trace("w",
                                   lambda *args: self.update_temp_settings("piece_style", self.piece_style_var.get()))
        style_menu = ttk.OptionMenu(style_frame, self.piece_style_var, self.temp_settings.piece_style,
                                    "circle", "outlined_circle", "square")
        style_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        style_frame.grid_columnconfigure(0, weight=1)
        style_menu["menu"].config(font=("Arial", 10))

        color_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["color_settings"], padding=10)
        color_frame.pack(fill="x", pady=5)

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["player_piece_color"]).grid(row=0, column=0,
                                                                                                  padx=5, pady=5,
                                                                                                  sticky=anchor)
        self.player_1_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_1_color))
        self.player_1_color_var.trace("w", lambda *args: self.update_temp_settings("player_1_color", hex_to_rgb(
            self.player_1_color_var.get())))
        self.p1_color_button = ttk.Button(color_frame, text="    ",
                                          command=lambda: self.choose_color(self.player_1_color_var,
                                                                            self.p1_color_button),
                                          style="Custom.TButton")
        self.p1_color_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["ai_piece_color"]).grid(row=1, column=0, padx=5,
                                                                                              pady=5, sticky=anchor)
        self.player_2_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_2_color))
        self.player_2_color_var.trace("w", lambda *args: self.update_temp_settings("player_2_color", hex_to_rgb(
            self.player_2_color_var.get())))
        self.p2_color_button = ttk.Button(color_frame, text="    ",
                                          command=lambda: self.choose_color(self.player_2_color_var,
                                                                            self.p2_color_button),
                                          style="Custom.TButton")
        self.p2_color_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["board_color_1"]).grid(row=2, column=0, padx=5,
                                                                                             pady=5, sticky=anchor)
        self.board_color_1_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_1))
        self.board_color_1_var.trace("w", lambda *args: self.update_temp_settings("board_color_1", hex_to_rgb(
            self.board_color_1_var.get())))
        self.b1_color_button = ttk.Button(color_frame, text="    ",
                                          command=lambda: self.choose_color(self.board_color_1_var,
                                                                            self.b1_color_button),
                                          style="Custom.TButton")
        self.b1_color_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["board_color_2"]).grid(row=3, column=0, padx=5,
                                                                                             pady=5, sticky=anchor)
        self.board_color_2_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_2))
        self.board_color_2_var.trace("w", lambda *args: self.update_temp_settings("board_color_2", hex_to_rgb(
            self.board_color_2_var.get())))
        self.b2_color_button = ttk.Button(color_frame, text="    ",
                                          command=lambda: self.choose_color(self.board_color_2_var,
                                                                            self.b2_color_button),
                                          style="Custom.TButton")
        self.b2_color_button.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())

        piece_image_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["piece_images"],
                                           padding=10)
        piece_image_frame.pack(fill="x", pady=5)

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_1_piece"]).grid(row=0, column=0,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_piece_image("player_1_piece"),
                   style="Custom.TButton").grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_1_king"]).grid(row=1, column=0,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_piece_image("player_1_king"),
                   style="Custom.TButton").grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_2_piece"]).grid(row=2, column=0,
                                                                                                    padx=5, pady=5,
                                                                                                    sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_piece_image("player_2_piece"),
                   style="Custom.TButton").grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_2_king"]).grid(row=3, column=0,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_piece_image("player_2_king"),
                   style="Custom.TButton").grid(row=3, column=1, padx=5, pady=5, sticky="w")

        sound_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["sound_settings"], padding=10)
        sound_frame.pack(fill="x", pady=5)
        self.sound_var = tk.BooleanVar(value=self.temp_settings.sound_enabled)
        self.sound_var.trace("w", lambda *args: self.update_temp_settings("sound_enabled", self.sound_var.get()))
        ttk.Radiobutton(sound_frame, text=LANGUAGES[self.settings.language]["sound_on"],
                        variable=self.sound_var, value=True).pack(side=side, padx=10)
        ttk.Radiobutton(sound_frame, text=LANGUAGES[self.settings.language]["sound_off"],
                        variable=self.sound_var, value=False).pack(side=side, padx=10)

    def search_ai_modules(self):
        """جستجوی خودکار ماژول‌های AI در دایرکتوری پروژه."""
        project_dir = Path(__file__).parent
        root_dir = project_dir.parent
        if str(root_dir) not in sys.path:
            sys.path.append(str(root_dir))

        modules = []
        ai_module_candidates = ["advanced_ai", "alphazero_ai", "minimax_ai"]

        for py_file in project_dir.glob("*.py"):
            module_name = py_file.stem
            if module_name.startswith("__") or module_name not in ai_module_candidates:
                continue
            full_module_name = f"a.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and name.endswith("AI"):
                        try:
                            metadata = obj.get_metadata()
                            default_type = metadata.get("default_type", module_name)
                            default_description = metadata.get("default_description", "")
                            modules.append({
                                "display": f"{module_name}:{name}",
                                "default_type": default_type,
                                "default_description": default_description,
                                "compatible": True
                            })
                        except AttributeError:
                            modules.append({
                                "display": f"{module_name}:{name}",
                                "default_type": module_name,
                                "default_description": "ناسازگار: متد get_metadata وجود ندارد",
                                "compatible": False
                            })
                        except Exception as e:
                            modules.append({
                                "display": f"{module_name}:{name}",
                                "default_type": module_name,
                                "default_description": f"ناسازگار: خطا در بارگذاری ({str(e)})",
                                "compatible": False
                            })
            except Exception as e:
                print(f"خطا در بارگذاری ماژول {full_module_name}: {e}")

        self.ai_module_class_combo["values"] = [m["display"] for m in modules] if modules else ["هیچ ماژولی پیدا نشد"]
        if modules:
            self.ai_module_class_var.set(modules[0]["display"])
            self.new_ai_type.set(modules[0]["default_type"])
            self.new_description.set(modules[0]["default_description"])
        else:
            self.ai_module_class_var.set("")
            self.new_ai_type.set("")
            self.new_description.set("")

        self.ai_modules = modules

    def update_ai_fields(self, event=None):
        """به‌روزرسانی فیلدهای نوع و توضیحات بر اساس ماژول انتخاب‌شده."""
        selected = self.ai_module_class_var.get()
        for module in getattr(self, "ai_modules", []):
            if module["display"] == selected:
                self.new_ai_type.set(module["default_type"])
                self.new_description.set(module["default_description"])
                if not module["compatible"]:
                    self.ai_module_class_combo.configure(foreground="red")  # هشدار بصری
                else:
                    self.ai_module_class_combo.configure(foreground="black")
                break
        else:
            self.new_ai_type.set("")
            self.new_description.set("")
            self.ai_module_class_combo.configure(foreground="black"

    def remove_selected_ai(self):
        """حذف AI انتخاب‌شده از لیست"""
        selected = self.ai_list.selection()
        if not selected:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["select_ai_to_remove"]
            )
            return
        ai_type = self.ai_list.item(selected[0])["values"][0]
        if ai_type == "none":
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["select_ai_to_remove"]
            )
            return
        if not messagebox.askyesno(
                LANGUAGES[self.settings.language]["warning"],
                f"آیا مطمئن هستید که می‌خواهید AI {ai_type} را حذف کنید؟"
        ):
            return
        config = load_config()
        if ai_type not in config["ai_types"]:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"نوع AI {ai_type} وجود ندارد"
            )
            return
        # حذف فایل‌های مرتبط
        try:
            pth_dir = Path(__file__).parent.parent / "pth"  # C:\Users\Arash W\desktop\learn\old\pth
            for player in ["player_1", "player_2"]:
                model_path = pth_dir / f"{ai_type}_{player}.pth"
                backup_path = pth_dir / f"backup_{ai_type}_{player}.pth"
                memory_path = pth_dir / f"long_term_memory_{player}.json"
                reward_path = pth_dir / f"episode_rewards_{player}.json"
                for path in [model_path, backup_path, memory_path, reward_path]:
                    if path.exists():
                        path.unlink()
        except Exception as e:
            messagebox.showwarning(
                LANGUAGES[self.settings.language]["warning"],
                f"خطا در حذف فایل‌های مرتبط با AI: {str(e)}"
            )
        # حذف از کانفیگ
        del config["ai_types"][ai_type]
        save_config(config)
        self.ai_types.remove(ai_type)
        self.player_1_ai_menu["menu"].delete(0, "end")
        self.player_2_ai_menu["menu"].delete(0, "end")
        for ai in self.ai_types:
            self.player_1_ai_menu["menu"].add_command(label=ai, command=lambda x=ai: self.player_1_ai_type_var.set(x))
            self.player_2_ai_menu["menu"].add_command(label=ai, command=lambda x=ai: self.player_2_ai_type_var.set(x))
        self.update_ai_list()
        messagebox.showinfo(
            LANGUAGES[self.settings.language]["厘米"],
            LANGUAGES[self.settings.language]["ai_removed"]
        )

    def update_ai_list(self):
        """به‌روزرسانی لیست AI‌های موجود"""
        for item in self.ai_list.get_children():
            self.ai_list.delete(item)
        config = load_ai_config()
        for ai_type, ai_info in config.get("ai_types", {}).items():
            status = "جدید" if ai_type == getattr(self, "newly_added_ai", None) else (
                "فعال" if ai_type in [self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"),
                                      self.temp_settings.ai_configs.get("player_2", {}).get("ai_type",
                                                                                            "none")] else "غیرفعال"
            )
            self.ai_list.insert("", "end", values=(ai_type, ai_info.get("description", ""), status))
        self.newly_added_ai = None

        # به‌روزرسانی ai_types و منوهای کشویی
        ai_config = load_ai_config()
        self.ai_types = ["none"] + [ai_type for ai_type in ai_config.get("ai_types", {}).keys()]
        self.player_1_ai_menu["menu"].delete(0, "end")
        self.player_2_ai_menu["menu"].delete(0, "end")
        for ai in self.ai_types:
            self.player_1_ai_menu["menu"].add_command(label=ai, command=lambda x=ai: self.player_1_ai_type_var.set(x))
            self.player_2_ai_menu["menu"].add_command(label=ai, command=lambda x=ai: self.player_2_ai_type_var.set(x))

    def setup_ai_tab(self, notebook):
        ai_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ai_frame, text=LANGUAGES[self.settings.language]["ai_settings_tab"])

        # جدول برای نمایش AI‌های موجود
        self.ai_list = ttk.Treeview(ai_frame, columns=("Type", "Description", "Status"), show="headings", height=4)
        self.ai_list.heading("Type", text=LANGUAGES[self.settings.language]["ai_type"])
        self.ai_list.heading("Description", text=LANGUAGES[self.settings.language]["description"])
        self.ai_list.heading("Status", text=LANGUAGES[self.settings.language]["status"])
        self.ai_list.pack(fill="both", expand=False, pady=5)

        ttk.Button(ai_frame, text=LANGUAGES[self.settings.language]["remove_selected"],
                   command=self.remove_selected_ai, style="Custom.TButton").pack(pady=5)

        form_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["add_new_ai"], padding=10)
        form_frame.pack(fill="x", pady=5)

        ttk.Label(form_frame, text=LANGUAGES[self.settings.language]["select_module_class"]).pack(anchor="w")
        self.ai_module_class_var = tk.StringVar()
        self.ai_module_class_combo = ttk.Combobox(form_frame, textvariable=self.ai_module_class_var, state="readonly")
        self.ai_module_class_combo.pack(fill="x", pady=5)
        self.ai_module_class_combo.bind("<<ComboboxSelected>>", self.update_ai_fields)

        ttk.Label(form_frame, text=LANGUAGES[self.settings.language]["new_ai_type"]).pack(anchor="w")
        self.new_ai_type = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.new_ai_type).pack(fill="x", pady=5)

        ttk.Label(form_frame, text=LANGUAGES[self.settings.language]["new_description"]).pack(anchor="w")
        self.new_description = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.new_description).pack(fill="x", pady=5)

        button_container = ttk.Frame(form_frame)
        button_container.pack(fill="x", pady=5)

        ttk.Button(button_container, text=LANGUAGES[self.settings.language]["search_modules"],
                   command=self.search_ai_modules, style="Custom.TButton").pack(side="left", padx=5)
        ttk.Button(button_container, text=LANGUAGES[self.settings.language]["add_ai"],
                   command=self.add_new_ai, style="Custom.TButton").pack(side="left", padx=5)

        # بخش AI Players
        ai_players_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["ai_players"], padding=10)
        ai_players_frame.pack(fill="x", pady=5)

        # بارگذاری ai_types
        ai_config = load_ai_config()
        self.ai_types = ["none"] + [ai_type for ai_type in ai_config.get("ai_types", {}).keys()]

        # Player 1: AI Type و Ability
        player_1_container = ttk.Frame(ai_players_frame)
        player_1_container.pack(fill="x", pady=2)

        ttk.Label(player_1_container, text=LANGUAGES[self.settings.language]["player_1_ai"]).pack(side="left", padx=5)
        self.player_1_ai_type_var = tk.StringVar(
            value=self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"))
        self.player_1_ai_menu = ttk.OptionMenu(player_1_container, self.player_1_ai_type_var,
                                               self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"),
                                               *self.ai_types)
        self.player_1_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_1_ai_menu["menu"].config(font=("Arial", 10))
        self.player_1_ai_type_var.trace("w", lambda *args: self.update_temp_settings("player_1_ai_type",
                                                                                     self.player_1_ai_type_var.get()))

        ttk.Label(player_1_container, text=LANGUAGES[self.settings.language]["ability"]).pack(side="left", padx=5)
        self.player_1_ability_var = tk.StringVar(value="medium")
        self.player_1_ability_menu = ttk.OptionMenu(
            player_1_container, self.player_1_ability_var, "medium",
            LANGUAGES[self.settings.language]["very_weak"],
            LANGUAGES[self.settings.language]["weak"],
            LANGUAGES[self.settings.language]["medium"],
            LANGUAGES[self.settings.language]["strong"],
            LANGUAGES[self.settings.language]["very_strong"],
            command=lambda _: self.update_ability_levels("player_1")
        )
        self.player_1_ability_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_1_ability_menu["menu"].config(font=("Arial", 10))

        # Player 2: AI Type و Ability
        player_2_container = ttk.Frame(ai_players_frame)
        player_2_container.pack(fill="x", pady=2)

        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["player_2_ai"]).pack(side="left", padx=5)
        self.player_2_ai_type_var = tk.StringVar(
            value=self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none"))
        self.player_2_ai_menu = ttk.OptionMenu(player_2_container, self.player_2_ai_type_var,
                                               self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none"),
                                               *self.ai_types)
        self.player_2_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_2_ai_menu["menu"].config(font=("Arial", 10))
        self.player_2_ai_type_var.trace("w", lambda *args: self.update_temp_settings("player_2_ai_type",
                                                                                     self.player_2_ai_type_var.get()))

        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["ability"]).pack(side="left", padx=5)
        self.player_2_ability_var = tk.StringVar(value="medium")
        self.player_2_ability_menu = ttk.OptionMenu(
            player_2_container, self.player_2_ability_var, "medium",
            LANGUAGES[self.settings.language]["very_weak"],
            LANGUAGES[self.settings.language]["weak"],
            LANGUAGES[self.settings.language]["medium"],
            LANGUAGES[self.settings.language]["strong"],
            LANGUAGES[self.settings.language]["very_strong"],
            command=lambda _: self.update_ability_levels("player_2")
        )
        self.player_2_ability_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_2_ability_menu["menu"].config(font=("Arial", 10))

        # بخش AI Pause Time
        pause_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["ai_pause_time"], padding=10)
        pause_frame.pack(fill="x", pady=5)
        ttk.Label(pause_frame, text=LANGUAGES[self.settings.language]["ai_pause_time_ms"]).pack(anchor="w")
        self.ai_pause_var = tk.IntVar(value=self.temp_settings.ai_pause_time)
        self.ai_pause_var.trace("w", lambda *args: self.update_temp_settings("ai_pause_time", self.ai_pause_var.get()))
        ttk.Entry(pause_frame, textvariable=self.ai_pause_var, width=10).pack(anchor="w", pady=5)

        # به‌روزرسانی لیست AI‌ها
        self.update_ai_list()
        self.search_ai_modules()

    def setup_player_tab(self, notebook):
        player_frame = ttk.Frame(notebook, padding="10")
        notebook.add(player_frame, text=LANGUAGES[self.settings.language]["player_tab"])

        side = "right" if self.settings.language == "fa" else "left"
        anchor = "e" if side == "left" else "w"

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_1_name"]).grid(row=0, column=0, padx=5, pady=5, sticky=anchor)
        self.player_1_name_var = tk.StringVar(value=self.temp_settings.player_1_name)
        ttk.Entry(player_frame, textvariable=self.player_1_name_var, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("player_1"), style="Custom.TButton").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_2_name"]).grid(row=1, column=0, padx=5, pady=5, sticky=anchor)
        self.player_2_name_var = tk.StringVar(value=self.temp_settings.player_2_name)
        ttk.Entry(player_frame, textvariable=self.player_2_name_var, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("player_2"), style="Custom.TButton").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["al1_name"]).grid(row=2, column=0, padx=5, pady=5, sticky=anchor)
        self.al1_name_var = tk.StringVar(value=self.temp_settings.al1_name)
        ttk.Entry(player_frame, textvariable=self.al1_name_var, width=15).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("al1"), style="Custom.TButton").grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["al2_name"]).grid(row=3, column=0, padx=5, pady=5, sticky=anchor)
        self.al2_name_var = tk.StringVar(value=self.temp_settings.al2_name)
        ttk.Entry(player_frame, textvariable=self.al2_name_var, width=15).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"],
                   command=lambda: self.upload_image("al2"), style="Custom.TButton").grid(row=3, column=2, padx=5, pady=5, sticky="w")

    def setup_advanced_tab(self, notebook):
        """Setup the Advanced Config tab with a button to open the advanced settings window."""
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_tab"])

        # هشدار تنظیمات پیشرفته
        ttk.Label(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_warning"],
                  wraplength=450, justify="center").pack(pady=10)

        # دکمه برای باز کردن پنجره تنظیمات پیشرفته
        ttk.Button(advanced_frame, text=LANGUAGES[self.settings.language]["open_advanced_config"],
                   command=self.open_advanced_config_window,
                   style="Custom.TButton").pack(pady=10)

    def open_advanced_config_window(self):
        """باز کردن پنجره تنظیمات پیشرفته."""
        AdvancedConfigWindow(self, self.settings, self.temp_settings, self.settings.language)

    def update_ability_levels(self, player):
        """Update AI ability level for the specified player."""
        ability_levels = {
            LANGUAGES[self.settings.language]["very_weak"]: 1,
            LANGUAGES[self.settings.language]["weak"]: 3,
            LANGUAGES[self.settings.language]["medium"]: 5,
            LANGUAGES[self.settings.language]["strong"]: 7,
            LANGUAGES[self.settings.language]["very_strong"]: 9
        }
        if player == "player_1":
            selected_level = self.player_1_ability_var.get()
            self.temp_settings.player_1_ability = ability_levels.get(selected_level, 5)
        elif player == "player_2":
            selected_level = self.player_2_ability_var.get()
            self.temp_settings.player_2_ability = ability_levels.get(selected_level, 5)

    def toggle_ai_vs_ai_options(self):
        """Toggle AI vs AI options based on game mode."""
        if self.play_with_var.get() == "ai_vs_ai":
            self.repeat_once_rb.config(state="normal")
            self.repeat_until_rb.config(state="normal")
        else:
            self.repeat_once_rb.config(state="disabled")
            self.repeat_until_rb.config(state="disabled")
            self.repeat_hands_frame.pack_forget()

    def toggle_repeat_options(self):
        """Toggle repeat options for AI vs AI mode."""
        if self.ai_vs_ai_var.get() == "repeat_game" and self.play_with_var.get() == "ai_vs_ai":
            self.repeat_hands_frame.pack(side="left", padx=5)
        else:
            self.repeat_hands_frame.pack_forget()

    def toggle_timer(self):
        """Toggle timer options."""
        if self.timer_var.get() == "with_timer":
            self.timer_duration_frame.pack(side="left", padx=5)
        else:
            self.timer_duration_frame.pack_forget()

    def choose_color(self, color_var, button):
        """Open color chooser and update button color."""
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            color_var.set(color)
            self.update_color_button(button, color)

    def update_color_button(self, button, color):
        """Update button background color."""
        try:
            button.configure(style=f"Color_{id(button)}.TButton")
            style = ttk.Style()
            style.configure(f"Color_{id(button)}.TButton", background=color)
        except ValueError:
            pass

    def upload_image(self, player):
        """Upload image for player profile."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            if player == "player_1":
                self.temp_settings.player_1_image = file_path
            elif player == "player_2":
                self.temp_settings.player_2_image = file_path
            elif player == "al1":
                self.temp_settings.al1_image = file_path
            elif player == "al2":
                self.temp_settings.al2_image = file_path

    def upload_piece_image(self, piece_type):
        """Upload image for piece type."""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            if piece_type == "player_1_piece":
                self.temp_settings.player_1_piece_image = file_path
            elif piece_type == "player_1_king":
                self.temp_settings.player_1_king_image = file_path
            elif piece_type == "player_2_piece":
                self.temp_settings.player_2_piece_image = file_path
            elif piece_type == "player_2_king":
                self.temp_settings.player_2_king_image = file_path

    def save(self):
        try:
            pause_time = self.ai_pause_var.get() if hasattr(self, 'ai_pause_var') else self.temp_settings.ai_pause_time
            if not 0 <= pause_time <= 5000:
                messagebox.showerror(LANGUAGES[self.settings.language]["error"],
                                     LANGUAGES[self.settings.language]["ai_pause_error"])
                return

            repeat_hands = self.hmh_var.get()
            if self.ai_vs_ai_var.get() == "repeat_game" and not 1 <= repeat_hands <= 1000:
                messagebox.showerror(LANGUAGES[self.settings.language]["error"],
                                     LANGUAGES[self.settings.language]["invalid_number_hands"])
                return

            # اعتبارسنجی نوع AI
            for ai_type, var in [(self.player_1_ai_type_var.get(), "AI بازیکن ۱"),
                                 (self.player_2_ai_type_var.get(), "AI بازیکن ۲")]:
                if ai_type != "none" and not self.check_ai_module(ai_type):
                    messagebox.showerror(
                        LANGUAGES[self.settings.language]["error"],
                        f"ماژول {var} با نوع '{ai_type}' پیدا نشد یا قابل بارگذاری نیست"
                    )
                    return

            config = {
                "player_starts": self.temp_settings.player_starts,
                "use_timer": self.temp_settings.use_timer,
                "game_time": self.temp_settings.game_time,
                "language": self.temp_settings.language,
                "player_1_color": rgb_to_hex(self.temp_settings.player_1_color),
                "player_2_color": rgb_to_hex(self.temp_settings.player_2_color),
                "board_color_1": rgb_to_hex(self.temp_settings.board_color_1),
                "board_color_2": rgb_to_hex(self.temp_settings.board_color_2),
                "piece_style": self.temp_settings.piece_style,
                "sound_enabled": self.temp_settings.sound_enabled,
                "ai_pause_time": pause_time,
                "game_mode": self.temp_settings.game_mode,
                "ai_vs_ai_mode": self.temp_settings.ai_vs_ai_mode,
                "repeat_hands": self.temp_settings.repeat_hands,
                "player_1_name": self.player_1_name_var.get() or "Player 1",
                "player_2_name": self.player_2_name_var.get() or "Player 2",
                "al1_name": self.al1_name_var.get() or "AI 1",
                "al2_name": self.al2_name_var.get() or "AI 2",
                "player_1_image": self.temp_settings.player_1_image or "",
                "player_2_image": self.temp_settings.player_2_image or "",
                "al1_image": self.temp_settings.al1_image or "",
                "al2_image": self.temp_settings.al2_image or "",
                "player_1_piece_image": self.temp_settings.player_1_piece_image or "",
                "player_1_king_image": self.temp_settings.player_1_king_image or "",
                "player_2_piece_image": self.temp_settings.player_2_piece_image or "",
                "player_2_king_image": self.temp_settings.player_2_king_image or "",
                "pause_between_hands": self.temp_settings.pause_between_hands,
                "player_1_ai_type": self.temp_settings.player_1_ai_type,
                "player_2_ai_type": self.temp_settings.player_2_ai_type,
                "player_1_ability": self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5),
                "player_2_ability": self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5),
                "ai_configs": self.temp_settings.ai_configs
            }

            save_config(config)

            for key, value in config.items():
                if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                    value = hex_to_rgb(value)
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                if hasattr(self.temp_settings, key):
                    setattr(self.temp_settings, key, value)

            if config["language"] != self.interface.settings.language:
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["warning"],
                    LANGUAGES[self.settings.language]["restart_required"]
                )

            self.interface.pending_settings = config
            if not self.interface.game.game_started or self.interface.game.game_over:
                self.interface.apply_pending_settings()
            else:
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["warning"],
                    LANGUAGES[self.settings.language]["apply_after_game"]
                )

            self.close()
        except tk.TclError:
            self.close()
        except json.JSONDecodeError:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                "فرمت JSON در پارامترهای آموزشی یا وزن‌های پاداش نامعتبر است"
            )

    def reset(self):
        if messagebox.askyesno(LANGUAGES[self.settings.language]["warning"],
                               LANGUAGES[self.settings.language]["reset_settings_warning"]):
            try:
                default_config = load_config()
                # بازنشانی temp_settings
                self.temp_settings = GameSettings()
                for key, value in default_config.items():
                    if hasattr(self.temp_settings, key):
                        if isinstance(value, dict):
                            getattr(self.temp_settings, key).update(value)
                        else:
                            setattr(self.temp_settings, key, value)

                # بازنشانی تنظیمات AI به صورت پویا
                self.temp_settings.ai_configs = {
                    "player_1": {
                        "ai_type": "none",
                        "ability_level": 5,
                        "training_params": default_config.get("default_ai_params", {}).get("training_params", {}),
                        "reward_weights": default_config.get("default_ai_params", {}).get("reward_weights", {})
                    },
                    "player_2": {
                        "ai_type": "none",
                        "ability_level": 5,
                        "training_params": default_config.get("default_ai_params", {}).get("training_params", {}),
                        "reward_weights": default_config.get("default_ai_params", {}).get("reward_weights", {})
                    }
                }

                # ایجاد یا به‌روزرسانی متغیرهای پیشرفته
                if not hasattr(self, 'player_1_training_params'):
                    self.player_1_training_params = tk.StringVar()
                self.player_1_training_params.set(
                    json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("training_params", {})))

                if not hasattr(self, 'player_1_reward_weights'):
                    self.player_1_reward_weights = tk.StringVar()
                self.player_1_reward_weights.set(
                    json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("reward_weights", {})))

                if not hasattr(self, 'player_2_training_params'):
                    self.player_2_training_params = tk.StringVar()
                self.player_2_training_params.set(
                    json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("training_params", {})))

                if not hasattr(self, 'player_2_reward_weights'):
                    self.player_2_reward_weights = tk.StringVar()
                self.player_2_reward_weights.set(
                    json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("reward_weights", {})))

                # به‌روزرسانی توانایی‌ها
                ability_mapping = {
                    1: "very_weak",
                    3: "weak",
                    5: "medium",
                    7: "strong",
                    9: "very_strong"
                }
                if hasattr(self, 'player_1_ability_var'):
                    self.player_1_ability_var.set(
                        LANGUAGES[self.settings.language][
                            ability_mapping.get(
                                self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5), "medium")
                        ]
                    )
                if hasattr(self, 'player_2_ability_var'):
                    self.player_2_ability_var.set(
                        LANGUAGES[self.settings.language][
                            ability_mapping.get(
                                self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5), "medium")
                        ]
                    )

                # به‌روزرسانی رابط کاربری
                self.update_ui()
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["info"],
                    LANGUAGES[self.settings.language]["settings_reset"]
                )
            except Exception as e:
                messagebox.showerror(
                    LANGUAGES[self.settings.language]["error"],
                    f"خطا در بازنشانی تنظیمات: {str(e)}"
                )

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
        if hasattr(self, 'ai_pause_var'):
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

        # به‌روزرسانی متغیرهای پیشرفته
        if hasattr(self, 'player_1_training_params'):
            self.player_1_training_params.set(
                json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("training_params", {})))
        if hasattr(self, 'player_1_reward_weights'):
            self.player_1_reward_weights.set(
                json.dumps(self.temp_settings.ai_configs.get("player_1", {}).get("reward_weights", {})))
        if hasattr(self, 'player_2_training_params'):
            self.player_2_training_params.set(
                json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("training_params", {})))
        if hasattr(self, 'player_2_reward_weights'):
            self.player_2_reward_weights.set(
                json.dumps(self.temp_settings.ai_configs.get("player_2", {}).get("reward_weights", {})))

        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())
        self.toggle_ai_vs_ai_options()
        self.toggle_repeat_options()
        self.toggle_timer()

class AdvancedConfigWindow(tk.Toplevel):
    """پنجره تنظیمات پیشرفته برای AI با تب‌های جداگانه برای هر بازیکن."""

    def __init__(self, parent, settings, temp_settings, language):
        super().__init__(parent.window)
        self.parent = parent
        self.settings = settings
        self.temp_settings = temp_settings
        self.title(LANGUAGES[language]["advanced_settings_title"])
        self.geometry("600x500")
        self.resizable(False, False)

        # نوت‌بوک برای تب‌های بازیکن‌ها
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # تب برای بازیکن 1
        self.setup_player_tab(notebook, "player_1", LANGUAGES[language]["player_1"])
        # تب برای بازیکن 2
        self.setup_player_tab(notebook, "player_2", LANGUAGES[language]["player_2"])

        # دکمه ذخیره
        ttk.Button(self, text=LANGUAGES[language]["save_all"],
                   command=self.save_advanced_settings,
                   style="Custom.TButton").pack(pady=10)

    def setup_player_tab(self, notebook, player_key, player_name):
        """ایجاد تب برای یک بازیکن."""
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text=player_name)

        # پارامترهای آموزش
        training_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.settings.language]["training_params"], padding=10)
        training_frame.pack(fill="x", pady=5)

        training_params = self.temp_settings.ai_configs.get(player_key, {}).get("training_params", {})

        # Learning Rate
        learning_rate_var = tk.DoubleVar(value=training_params.get("learning_rate", 0.001))
        ttk.Label(training_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=learning_rate_var).grid(row=0, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_learning_rate_var", learning_rate_var)

        # Gamma
        gamma_var = tk.DoubleVar(value=training_params.get("gamma", 0.99))
        ttk.Label(training_frame, text="Gamma:").grid(row=1, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=gamma_var).grid(row=1, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_gamma_var", gamma_var)

        # Batch Size
        batch_size_var = tk.IntVar(value=training_params.get("batch_size", 128))
        ttk.Label(training_frame, text="Batch Size:").grid(row=2, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=batch_size_var).grid(row=2, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_batch_size_var", batch_size_var)

        # Memory Size
        memory_size_var = tk.IntVar(value=training_params.get("memory_size", 10000))
        ttk.Label(training_frame, text="Memory Size:").grid(row=3, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=memory_size_var).grid(row=3, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_memory_size_var", memory_size_var)

        # وزن‌های پاداش
        reward_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.settings.language]["reward_weights"], padding=10)
        reward_frame.pack(fill="x", pady=5)

        reward_weights = self.temp_settings.ai_configs.get(player_key, {}).get("reward_weights", {})

        # Piece Difference
        piece_difference_var = tk.DoubleVar(value=reward_weights.get("piece_difference", 1.0))
        ttk.Label(reward_frame, text="Piece Difference:").grid(row=0, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=piece_difference_var).grid(row=0, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_piece_difference_var", piece_difference_var)

        # King Bonus
        king_bonus_var = tk.DoubleVar(value=reward_weights.get("king_bonus", 2.0))
        ttk.Label(reward_frame, text="King Bonus:").grid(row=1, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=king_bonus_var).grid(row=1, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_king_bonus_var", king_bonus_var)

        # Position Bonus
        position_bonus_var = tk.DoubleVar(value=reward_weights.get("position_bonus", 0.1))
        ttk.Label(reward_frame, text="Position Bonus:").grid(row=2, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=position_bonus_var).grid(row=2, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_position_bonus_var", position_bonus_var)

        # Capture Bonus
        capture_bonus_var = tk.DoubleVar(value=reward_weights.get("capture_bonus", 1.0))
        ttk.Label(reward_frame, text="Capture Bonus:").grid(row=3, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=capture_bonus_var).grid(row=3, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_capture_bonus_var", capture_bonus_var)

        # Multi-Jump Bonus
        multi_jump_bonus_var = tk.DoubleVar(value=reward_weights.get("multi_jump_bonus", 2.0))
        ttk.Label(reward_frame, text="Multi-Jump Bonus:").grid(row=4, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=multi_jump_bonus_var).grid(row=4, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_multi_jump_bonus_var", multi_jump_bonus_var)

        # King Capture Bonus
        king_capture_bonus_var = tk.DoubleVar(value=reward_weights.get("king_capture_bonus", 3.0))
        ttk.Label(reward_frame, text="King Capture Bonus:").grid(row=5, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=king_capture_bonus_var).grid(row=5, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_king_capture_bonus_var", king_capture_bonus_var)

        # Mobility Bonus
        mobility_bonus_var = tk.DoubleVar(value=reward_weights.get("mobility_bonus", 0.1))
        ttk.Label(reward_frame, text="Mobility Bonus:").grid(row=6, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=mobility_bonus_var).grid(row=6, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_mobility_bonus_var", mobility_bonus_var)

        # Defensive Bonus
        defensive_bonus_var = tk.DoubleVar(value=reward_weights.get("defensive_bonus", 0.2))
        ttk.Label(reward_frame, text="Defensive Bonus:").grid(row=7, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=defensive_bonus_var).grid(row=7, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_defensive_bonus_var", defensive_bonus_var)

    def save_advanced_settings(self):
        """ذخیره تنظیمات پیشرفته برای هر دو بازیکن."""
        try:
            for player_key in ["player_1", "player_2"]:
                # به‌روزرسانی پارامترهای آموزشی
                training_params = {
                    "learning_rate": getattr(self, f"{player_key}_learning_rate_var").get(),
                    "gamma": getattr(self, f"{player_key}_gamma_var").get(),
                    "batch_size": getattr(self, f"{player_key}_batch_size_var").get(),
                    "memory_size": getattr(self, f"{player_key}_memory_size_var").get(),
                }
                # به‌روزرسانی وزن‌های پاداش
                reward_weights = {
                    "piece_difference": getattr(self, f"{player_key}_piece_difference_var").get(),
                    "king_bonus": getattr(self, f"{player_key}_king_bonus_var").get(),
                    "position_bonus": getattr(self, f"{player_key}_position_bonus_var").get(),
                    "capture_bonus": getattr(self, f"{player_key}_capture_bonus_var").get(),
                    "multi_jump_bonus": getattr(self, f"{player_key}_multi_jump_bonus_var").get(),
                    "king_capture_bonus": getattr(self, f"{player_key}_king_capture_bonus_var").get(),
                    "mobility_bonus": getattr(self, f"{player_key}_mobility_bonus_var").get(),
                    "defensive_bonus": getattr(self, f"{player_key}_defensive_bonus_var").get(),
                }
                # به‌روزرسانی ai_configs در temp_settings
                if player_key not in self.temp_settings.ai_configs:
                    self.temp_settings.ai_configs[player_key] = {}
                self.temp_settings.ai_configs[player_key].update({
                    "training_params": training_params,
                    "reward_weights": reward_weights
                })

            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["advanced_settings_saved"]
            )
            self.destroy()
        except tk.TclError as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در ذخیره تنظیمات پیشرفته: {str(e)}"
            )