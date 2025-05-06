import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from .game import Game
from .settings import GameSettings
from .config import save_config, load_config, load_ai_config, save_ai_config, LANGUAGES
from .utils import hex_to_rgb, rgb_to_hex
import torch
import os
import json
import sys
import importlib
from pathlib import Path

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

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

    def load_ai_config(self):
        """بارگذاری تنظیمات AI از ai_config.json با استفاده از تابع config.py"""
        ai_config = load_ai_config()
        self.ai_types = ["none"] + list(ai_config.get("ai_types", {}).keys())
        # به‌روزرسانی تنظیمات موقت با تنظیمات AI
        self.temp_settings.ai_configs = ai_config.get("ai_configs", {
            "player_1": {"ai_type": "none", "ability_level": 5, "training_params": {}, "reward_weights": {}},
            "player_2": {"ai_type": "none", "ability_level": 5, "training_params": {}, "reward_weights": {}}
        })

    def check_ai_module(self, ai_type):
        """بررسی وجود ماژول AI با استفاده از ai_config.json"""
        if ai_type == "none":
            return True
        ai_config = load_ai_config()
        print(f"check_ai_module: Checking ai_type '{ai_type}' in {ai_config['ai_types'].keys()}")
        return ai_type in ai_config["ai_types"]

    def add_ai_type(self, ai_type, module_name, class_name, description):
        """افزودن نوع AI جدید به ai_config.json"""
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
        project_dir = Path(__file__).parent
        module_path = project_dir / f"{module_name}.py"
        if not module_path.exists():
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"ماژول {module_name}.py پیدا نشد"
            )
            return
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
                f"خطا در بررسی ماژول: {str(e)}"
            )
            return

        ai_config = load_ai_config()
        ai_config["ai_types"][ai_type] = {
            "module": module_name,  # ذخیره نام ماژول بدون پیشوند
            "class": class_name,
            "description": description,
            "training_params": ai_config["default_ai_params"]["training_params"],
            "reward_weights": ai_config["default_ai_params"]["reward_weights"]
        }
        save_ai_config(ai_config)
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
        """حذف نوع AI از ai_config.json"""
        if ai_type == "none":
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                "Cannot remove 'none' type"
            )
            return
        if ai_type in [self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"),
                       self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none")]:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"AI '{ai_type}' در حال استفاده است و نمی‌تواند حذف شود"
            )
            return
        ai_config = load_ai_config()
        if ai_type in ai_config["ai_types"]:
            del ai_config["ai_types"][ai_type]
            save_ai_config(ai_config)
            self.ai_types.remove(ai_type)
            self.update_ai_dropdowns()
            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["ai_removed"]
            )
        else:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"AI type '{ai_type}' not found"
            )

    def update_ai_dropdowns(self):
        """بازسازی کامل منوهای کشویی انتخاب AI و به‌روزرسانی ai_config.json"""
        ai_config = load_ai_config()
        print(f"Updating AI dropdowns with ai_types: {ai_config['ai_types'].keys()}")

        # بازسازی منوهای بازیکن ۱ و ۲
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

        # منوی حذف AI (فقط اگر تعریف شده باشد)
        if hasattr(self, 'remove_ai_menu'):
            self.remove_ai_menu['menu'].delete(0, 'end')
            for ai_type in self.ai_types[1:]:  # نادیده گرفتن "none"
                self.remove_ai_menu['menu'].add_command(
                    label=ai_type,
                    command=lambda value=ai_type: self.remove_ai_var.set(value)
                )

        # تنظیم مقدار پیش‌فرض اگر AI انتخاب‌شده دیگر وجود ندارد
        if self.player_1_ai_type_var.get() not in self.ai_types:
            self.player_1_ai_type_var.set("none")
            self._update_player_ai_config("player_1", "none")
            print(f"Reset player_1 AI to 'none' (was {self.player_1_ai_type_var.get()})")
        if self.player_2_ai_type_var.get() not in self.ai_types:
            self.player_2_ai_type_var.set("none")
            self._update_player_ai_config("player_2", "none")
            print(f"Reset player_2 AI to 'none' (was {self.player_2_ai_type_var.get()})")

    def _update_player_ai_config(self, player, ai_type):
        """به‌روزرسانی temp_settings برای بازیکن با تنظیمات پیش‌فرض"""
        ai_config = load_ai_config()
        print(f"Updating {player} temp_settings to AI: {ai_type}")

        # اعتبارسنجی ai_type
        if ai_type != "none" and ai_type not in ai_config["ai_types"]:
            print(f"Error: AI type '{ai_type}' not found in ai_config['ai_types']")
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"ماژول AI {ai_type} برای {player} پیدا نشد"
            )
            self.player_1_ai_type_var.set("none") if player == "player_1" else self.player_2_ai_type_var.set("none")
            ai_type = "none"

        # ذخیره ai_type در temp_settings
        self.temp_settings.ai_configs[player]["ai_type"] = ai_type

        # اعمال تنظیمات پیش‌فرض در temp_settings
        default_params = ai_config.get("default_ai_params", {
            "ability_level": 5,
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
        })
        if ai_type != "none":
            self.temp_settings.ai_configs[player]["ability_level"] = default_params["ability_level"]
            self.temp_settings.ai_configs[player]["training_params"] = default_params["training_params"].copy()
            self.temp_settings.ai_configs[player]["reward_weights"] = default_params["reward_weights"].copy()
        else:
            self.temp_settings.ai_configs[player]["ability_level"] = default_params["ability_level"]
            self.temp_settings.ai_configs[player]["training_params"] = {}
            self.temp_settings.ai_configs[player]["reward_weights"] = {}

        print(f"Updated {player} temp_settings: {self.temp_settings.ai_configs[player]}")

    def add_new_ai(self):
        """افزودن AI جدید به ai_config.json فقط با اطلاعات اولیه"""
        ai_type = self.new_ai_type.get().strip()
        description = self.new_description.get().strip()
        module_name = self.ai_module_class_var.get().strip()

        # اعتبارسنجی ورودی‌ها
        if not ai_type or not module_name:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["fill_all_fields"]
            )
            print(f"خطا: ai_type={ai_type}, module_name={module_name}")
            return
        if ai_type in self.ai_types:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["ai_type_exists"]
            )
            print(f"خطا: ai_type '{ai_type}' قبلاً وجود دارد")
            return

        # پیدا کردن class_name از ai_modules
        class_name = None
        for module in self.ai_modules:
            if module["display"] == module_name:
                class_name = module.get("class_name")
                break
        if not class_name:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"ماژول {module_name} کلاس AI معتبری ندارد"
            )
            print(f"خطا: class_name برای {module_name} پیدا نشد")
            return

        full_module_name = f"a.{module_name}"
        print(f"تلاش برای بارگذاری ماژول: {full_module_name}, کلاس: {class_name}")

        # بررسی وجود ماژول و کلاس
        try:
            module = importlib.import_module(full_module_name)
            if not hasattr(module, class_name):
                messagebox.showerror(
                    LANGUAGES[self.settings.language]["error"],
                    f"کلاس {class_name} در ماژول {module_name} پیدا نشد"
                )
                print(f"خطا: کلاس {class_name} در {full_module_name} پیدا نشد")
                return
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در بارگذاری ماژول {module_name}: {str(e)}"
            )
            print(f"خطا در بارگذاری {full_module_name}: {str(e)}")
            return

        # بارگذاری و به‌روزرسانی ai_config
        ai_config = load_ai_config()
        ai_config["ai_types"][ai_type] = {
            "module": full_module_name,
            "class": class_name,
            "description": description
        }
        try:
            save_ai_config(ai_config)
            print(f"AI جدید ذخیره شد: {ai_type}")
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در ذخیره AI جدید: {str(e)}"
            )
            print(f"خطا در ذخیره ai_config: {str(e)}")
            return

        # به‌روزرسانی self.ai_types
        if not hasattr(self, 'ai_types') or not self.ai_types:
            self.ai_types = ["none"]
        if ai_type not in self.ai_types:
            self.ai_types.append(ai_type)
            print(f"AI {ai_type} به self.ai_types اضافه شد: {self.ai_types}")

        # به‌روزرسانی رابط کاربری
        try:
            self.update_ai_dropdowns()
            self.update_ai_list()
            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["ai_added"]
            )
            print(f"AI {ai_type} به منوی کشویی و لیست اضافه شد")
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در به‌روزرسانی رابط کاربری: {str(e)}"
            )
            print(f"خطا در به‌روزرسانی رابط کاربری: {str(e)}")

    def remove_selected_ai(self):
        """حذف AI انتخاب‌شده از ai_config.json"""
        selected = self.ai_list.selection()
        if not selected:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                LANGUAGES[self.settings.language]["no_ai_selected"]
            )
            return

        ai_type = self.ai_list.item(selected[0])["values"][0]
        if ai_type in [self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none"),
                       self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none")]:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"AI '{ai_type}' در حال استفاده است و نمی‌تواند حذف شود"
            )
            return

        if messagebox.askyesno(
                LANGUAGES[self.settings.language]["warning"],
                f"آیا مطمئن هستید که می‌خواهید AI '{ai_type}' را حذف کنید؟"
        ):
            ai_config = load_ai_config()
            if ai_type in ai_config["ai_types"]:
                del ai_config["ai_types"][ai_type]
                save_ai_config(ai_config)
                self.ai_types = ["none"] + [t for t in ai_config["ai_types"].keys()]
                self.update_ai_dropdowns()  # به‌روزرسانی منوهای کشویی
                self.update_ai_list()
                messagebox.showinfo(
                    LANGUAGES[self.settings.language]["info"],
                    LANGUAGES[self.settings.language]["ai_removed"]
                )
            else:
                messagebox.showerror(
                    LANGUAGES[self.settings.language]["error"],
                    f"AI '{ai_type}' پیدا نشد"
                )

    def copy_current_settings(self):
        """کپی تنظیمات فعلی به temp_settings"""
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
        """به‌روزرسانی temp_settings با مقدار جدید"""
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
            self.temp_settings.ai_configs["player_1"]["ai_type"] = value
            print(f"Updated temp_settings player_1_ai_type: {value}")
        elif key == "player_2_ai_type":
            self.temp_settings.ai_configs["player_2"]["ai_type"] = value
            print(f"Updated temp_settings player_2_ai_type: {value}")
        elif key == "ai_pause_time":
            self.temp_settings.ai_pause_time = value

    def setup_design_tab(self, notebook):
        design_frame = ttk.Frame(notebook, padding="10")
        notebook.add(design_frame, text=LANGUAGES[self.settings.language]["design_tab"])

        # چک‌باکس صدا
        sound_container = ttk.Frame(design_frame)
        sound_container.pack(fill="x", pady=2)
        ttk.Label(sound_container, text=LANGUAGES[self.settings.language]["sound_enabled"]).pack(side="left", padx=5)
        self.sound_var = tk.BooleanVar(value=self.temp_settings.sound_enabled)
        ttk.Checkbutton(sound_container, variable=self.sound_var,
                        command=lambda: self.update_temp_settings("sound_enabled", self.sound_var.get())).pack(
            side="left", padx=5)

        # Player 1 Color
        player_1_color_container = ttk.Frame(design_frame)
        player_1_color_container.pack(fill="x", pady=2)
        ttk.Label(player_1_color_container, text=LANGUAGES[self.settings.language]["player_1_color"]).pack(side="left",
                                                                                                           padx=5)
        try:
            self.player_1_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_1_color))
        except ValueError:
            self.player_1_color_var = tk.StringVar(value="#ff0000")
        self.p1_color_button = ttk.Button(player_1_color_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.player_1_color_var, "p1_color_button"))
        self.p1_color_button.pack(side="left", padx=5)
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())

        # Player 2 Color
        player_2_color_container = ttk.Frame(design_frame)
        player_2_color_container.pack(fill="x", pady=2)
        ttk.Label(player_2_color_container, text=LANGUAGES[self.settings.language]["player_2_color"]).pack(side="left",
                                                                                                           padx=5)
        try:
            self.player_2_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_2_color))
        except ValueError:
            self.player_2_color_var = tk.StringVar(value="#0000ff")
        self.p2_color_button = ttk.Button(player_2_color_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.player_2_color_var, "p2_color_button"))
        self.p2_color_button.pack(side="left", padx=5)
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())

        # Board Color 1
        board_color_1_container = ttk.Frame(design_frame)
        board_color_1_container.pack(fill="x", pady=2)
        ttk.Label(board_color_1_container, text=LANGUAGES[self.settings.language]["board_color_1"]).pack(side="left",
                                                                                                         padx=5)
        try:
            self.board_color_1_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_1))
        except ValueError:
            self.board_color_1_var = tk.StringVar(value="#ffffff")
        self.b1_color_button = ttk.Button(board_color_1_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.board_color_1_var, "b1_color_button"))
        self.b1_color_button.pack(side="left", padx=5)
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())

        # Board Color 2
        board_color_2_container = ttk.Frame(design_frame)
        board_color_2_container.pack(fill="x", pady=2)
        ttk.Label(board_color_2_container, text=LANGUAGES[self.settings.language]["board_color_2"]).pack(side="left",
                                                                                                         padx=5)
        try:
            self.board_color_2_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_2))
        except ValueError:
            self.board_color_2_var = tk.StringVar(value="#8b4513")
        self.b2_color_button = ttk.Button(board_color_2_container, text="Choose Color",
                                          command=lambda: self.choose_color(self.board_color_2_var, "b2_color_button"))
        self.b2_color_button.pack(side="left", padx=5)
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())

        # Piece Style
        piece_style_container = ttk.Frame(design_frame)
        piece_style_container.pack(fill="x", pady=2)
        ttk.Label(piece_style_container, text=LANGUAGES[self.settings.language]["piece_style"]).pack(side="left",
                                                                                                     padx=5)
        self.piece_style_var = tk.StringVar(value=self.temp_settings.piece_style)
        piece_styles = ["circle", "outlined_circle", "square", "diamond", "star", "custom"]
        ttk.OptionMenu(piece_style_container, self.piece_style_var, self.temp_settings.piece_style,
                       *piece_styles,
                       command=lambda _: self.update_temp_settings("piece_style", self.piece_style_var.get())).pack(
            side="left", fill="x", expand=True, padx=5)

        # پیش‌نمایش استایل مهره
        preview_container = ttk.Frame(design_frame)
        preview_container.pack(fill="x", pady=5)
        ttk.Label(preview_container, text="Piece Preview").pack(side="left", padx=5)
        self.preview_canvas = tk.Canvas(preview_container, width=50, height=50, bg="white")
        self.preview_canvas.pack(side="left", padx=5)
        self.update_piece_preview()

        # تصاویر سفارشی
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

    def upload_image(self, piece, var):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            var.set(file_path)
            self.update_temp_settings(f"{piece}_image", file_path)

    def save(self):
        self.temp_settings.player_1_color = hex_to_rgb(self.player_1_color_var.get())
        self.temp_settings.player_2_color = hex_to_rgb(self.player_2_color_var.get())
        self.temp_settings.board_color_1 = hex_to_rgb(self.board_color_1_var.get())
        self.temp_settings.board_color_2 = hex_to_rgb(self.board_color_2_var.get())
        self.temp_settings.piece_style = self.piece_style_var.get()
        self.temp_settings.sound_enabled = self.sound_var.get()
        for piece in ["player_1_piece", "player_1_king", "player_2_piece", "player_2_king"]:
            setattr(self.temp_settings, f"{piece}_image", self.entries[piece].get())
        self.interface.apply_pending_settings(self.temp_settings.__dict__)
        self.close()

    def search_ai_modules(self):
        """جستجوی خودکار ماژول‌های AI در دایرکتوری پروژه"""
        project_dir = Path(__file__).parent
        root_dir = project_dir.parent
        print(f"دایرکتوری پروژه: {project_dir}")
        print(f"ریشه پروژه: {root_dir}")

        # تنظیم sys.path
        if str(project_dir) not in sys.path:
            sys.path.append(str(project_dir))
        if str(root_dir) not in sys.path:
            sys.path.append(str(root_dir))
        print(f"sys.path: {sys.path}")

        modules = []
        print(f"فایل‌های .py در {project_dir}: {[f.name for f in project_dir.glob('*.py')]}")

        for py_file in project_dir.glob("*.py"):
            module_name = py_file.stem
            if module_name.startswith("__") or module_name == "base_ai" or not module_name.isidentifier():
                print(f"نادیده گرفتن فایل: {module_name}")
                continue
            full_module_name = f"a.{module_name}"
            print(f"تلاش برای بارگذاری ماژول: {full_module_name}")

            try:
                module = importlib.import_module(full_module_name)
                print(f"ماژول {full_module_name} با موفقیت بارگذاری شد")
                found_ai_class = False
                class_name = None
                for name, obj in module.__dict__.items():
                    if isinstance(obj, type) and name.endswith("AI") and name != "BaseAI":
                        class_name = name
                        found_ai_class = True
                        break  # فقط اولین کلاس AI معتبر رو می‌گیریم
                if not found_ai_class:
                    print(f"هیچ کلاس AI معتبری در {full_module_name} پیدا نشد")
                    continue

                # خواندن متادیتا از AI_METADATA
                try:
                    metadata = getattr(module, "AI_METADATA", None)
                    if not metadata:
                        print(f"متادیتا در {full_module_name} پیدا نشد، استفاده از مقادیر پیش‌فرض")
                        metadata = {
                            "default_type": module_name,
                            "default_description": f"هوش مصنوعی {module_name}"
                        }
                    default_type = metadata.get("default_type", module_name)
                    default_description = metadata.get("default_description", f"هوش مصنوعی {module_name}")
                    if not any(m["default_type"] == default_type for m in modules):
                        modules.append({
                            "display": module_name,
                            "default_type": default_type,
                            "default_description": default_description,
                            "class_name": class_name
                        })
                        print(f"ماژول شناسایی‌شده: {module_name}, کلاس: {class_name}, متادیتا: {metadata}")
                except Exception as e:
                    print(f"خطا در خواندن متادیتا برای {full_module_name}: {e}")
                    continue

            except SyntaxError as e:
                print(f"خطای سینتکس در {full_module_name}: {e} (خط {e.lineno})")
            except ImportError as e:
                print(f"خطا در بارگذاری ماژول {full_module_name}: {e}")
            except Exception as e:
                print(f"خطای غیرمنتظره در {full_module_name}: {e}")

        # تنظیم منوی کشویی
        self.ai_module_class_combo["values"] = [m["display"] for m in modules] if modules else ["هیچ ماژولی پیدا نشد"]
        if modules:
            self.ai_module_class_var.set(modules[0]["display"])
            self.new_ai_type.set(modules[0]["default_type"])
            self.new_description.set(modules[0]["default_description"])
            print(f"منوی کشویی تنظیم شد: {[m['display'] for m in modules]}")
        else:
            self.ai_module_class_var.set("")
            self.new_ai_type.set("")
            self.new_description.set("")
            print("هیچ ماژولی برای منوی کشویی پیدا نشد")
        self.ai_modules = modules
        print(f"ماژول‌های نهایی شناسایی‌شده: {[m['display'] for m in modules]}")

    def update_ai_fields(self, event=None):
        """به‌روزرسانی فیلدهای نوع و توضیحات بر اساس ماژول انتخاب‌شده"""
        selected = self.ai_module_class_var.get()
        for module in getattr(self, "ai_modules", []):
            if module["display"] == selected:
                self.new_ai_type.set(module["default_type"])
                self.new_description.set(module["default_description"])
                break
        else:
            self.new_ai_type.set("")
            self.new_description.set("")

    def update_ai_list(self):
        self.ai_list.delete(*self.ai_list.get_children())
        ai_config = load_ai_config()
        for ai_type in self.ai_types:
            if ai_type == "none":
                continue
            ai_info = ai_config["ai_types"].get(ai_type, {})
            description = ai_info.get("description", "")
            self.ai_list.insert("", "end", values=(ai_type, description))

    def setup_ai_tab(self, notebook):
        ai_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ai_frame, text=LANGUAGES[self.settings.language]["ai_settings_tab"])

        # جدول برای نمایش AI‌های موجود
        self.ai_list = ttk.Treeview(ai_frame, columns=("Type", "Description", "Status"), show="headings", height=4)
        self.ai_list.heading("Type", text=LANGUAGES[self.settings.language]["ai_type"])
        self.ai_list.heading("Description", text=LANGUAGES[self.settings.language]["description"])
        self.ai_list.heading("Status", text=LANGUAGES[self.settings.language]["status"])
        self.ai_list.pack(fill="both", expand=False, pady=5)

        # منوی کشویی برای حذف AI
        remove_container = ttk.Frame(ai_frame)
        remove_container.pack(fill="x", pady=5)
        ttk.Label(remove_container, text=LANGUAGES[self.settings.language]["remove_ai"]).pack(side="left", padx=5)
        self.remove_ai_var = tk.StringVar(value="")
        self.remove_ai_menu = ttk.OptionMenu(remove_container, self.remove_ai_var, "", *self.ai_types[1:])
        self.remove_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.remove_ai_menu["menu"].config(font=("Arial", 10))
        ttk.Button(ai_frame, text=LANGUAGES[self.settings.language]["remove_selected"],
                   command=self.remove_selected_ai, style="Custom.TButton").pack(pady=5)

        # فرم افزودن AI جدید
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

        # Player 1: AI Type و Ability
        player_1_container = ttk.Frame(ai_players_frame)
        player_1_container.pack(fill="x", pady=2)

        ttk.Label(player_1_container, text=LANGUAGES[self.settings.language]["player_1_ai"]).pack(side="left", padx=5)
        player_1_ai_type = self.temp_settings.ai_configs.get("player_1", {}).get("ai_type", "none")
        if player_1_ai_type not in self.ai_types:
            player_1_ai_type = "none"
        self.player_1_ai_type_var = tk.StringVar(value=player_1_ai_type)
        self.player_1_ai_menu = ttk.OptionMenu(player_1_container, self.player_1_ai_type_var,
                                               player_1_ai_type, *self.ai_types)
        self.player_1_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_1_ai_menu["menu"].config(font=("Arial", 10))
        self.player_1_ai_type_var.trace("w", lambda *args: self.update_temp_settings("player_1_ai_type",
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
        self.player_1_ability_var = tk.StringVar(
            value=LANGUAGES[self.settings.language][ability_mapping.get(player_1_ability, "medium")])
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

        # Player 2: AI Type و Ability
        player_2_container = ttk.Frame(ai_players_frame)
        player_2_container.pack(fill="x", pady=2)

        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["player_2_ai"]).pack(side="left", padx=5)
        player_2_ai_type = self.temp_settings.ai_configs.get("player_2", {}).get("ai_type", "none")
        if player_2_ai_type not in self.ai_types:
            player_2_ai_type = "none"
        self.player_2_ai_type_var = tk.StringVar(value=player_2_ai_type)
        self.player_2_ai_menu = ttk.OptionMenu(player_2_container, self.player_2_ai_type_var,
                                               player_2_ai_type, *self.ai_types)
        self.player_2_ai_menu.pack(side="left", fill="x", expand=True, padx=5)
        self.player_2_ai_menu["menu"].config(font=("Arial", 10))
        self.player_2_ai_type_var.trace("w", lambda *args: self.update_temp_settings("player_2_ai_type",
                                                                                     self.player_2_ai_type_var.get()))

        ttk.Label(player_2_container, text=LANGUAGES[self.settings.language]["ability"]).pack(side="left", padx=5)
        player_2_ability = self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5)
        self.player_2_ability_var = tk.StringVar(
            value=LANGUAGES[self.settings.language][ability_mapping.get(player_2_ability, "medium")])
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

        # بخش AI Pause Time
        pause_frame = ttk.LabelFrame(ai_frame, text=LANGUAGES[self.settings.language]["ai_pause_time"], padding=10)
        pause_frame.pack(fill="x", pady=5)
        ttk.Label(pause_frame, text=LANGUAGES[self.settings.language]["ai_pause_time_ms"]).pack(anchor="w")
        self.ai_pause_var = tk.IntVar(value=self.temp_settings.ai_pause_time)
        self.ai_pause_var.trace("w", lambda *args: self.update_temp_settings("ai_pause_time", self.ai_pause_var.get()))
        ttk.Entry(pause_frame, textvariable=self.ai_pause_var, width=10).pack(anchor="w", pady=5)

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
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_tab"])

        ttk.Label(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_warning"],
                  wraplength=450, justify="center").pack(pady=10)

        ttk.Button(advanced_frame, text=LANGUAGES[self.settings.language]["open_advanced_config"],
                   command=self.open_advanced_config_window,
                   style="Custom.TButton").pack(pady=10)

    def open_advanced_config_window(self):
        AdvancedConfigWindow(self, self.settings, self.temp_settings, self.settings.language)

    def save_advanced_settings(self):
        """ذخیره تنظیمات پیشرفته برای AI در ai_config.json"""
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
                "gamma": self.player2_gamma_var.get(),
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
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در ذخیره تنظیمات پیشرفته: {str(e)}"
            )

    def update_ability_levels(self, player):
        """به‌روزرسانی سطح توانایی AI برای بازیکن مشخص"""
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

    def choose_color(self, color_var, button):
        color = colorchooser.askcolor(title="Choose Color")[1]
        if color:
            color_var.set(color)
            self.update_color_button(button, color)

    def update_color_button(self, button, color):
        """به‌روزرسانی رنگ دکمه با استایل ttk"""
        try:
            style_name = f"Color_{id(button)}.TButton"
            style = ttk.Style()
            style.configure(style_name, background=color)
            button.configure(style=style_name)
        except Exception as e:
            import logging
            logging.error(f"Error updating button color: {str(e)}")

    def upload_image(self, player):
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
        """ذخیره تنظیمات در config.json و ai_config.json"""
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

            ai_config = load_ai_config()
            print(f"save: Loaded ai_config['ai_types']: {ai_config['ai_types'].keys()}")

            for ai_type, var in [(self.player_1_ai_type_var.get(), "AI بازیکن ۱"),
                                 (self.player_2_ai_type_var.get(), "AI بازیکن ۲")]:
                if ai_type != "none" and ai_type not in ai_config["ai_types"]:
                    print(f"Error: AI type '{ai_type}' not found in ai_config['ai_types']")
                    messagebox.showerror(
                        LANGUAGES[self.settings.language]["error"],
                        f"ماژول {var} با نوع '{ai_type}' پیدا نشد"
                    )
                    return

            config = load_config()
            config.update({
                "language": self.temp_settings.language,
                "game_mode": self.temp_settings.game_mode,
                "ai_vs_ai_mode": self.temp_settings.ai_vs_ai_mode,
                "repeat_hands": self.temp_settings.repeat_hands,
                "player_starts": self.temp_settings.player_starts,
                "use_timer": self.temp_settings.use_timer,
                "game_time": self.temp_settings.game_time,
                "piece_style": self.temp_settings.piece_style,
                "sound_enabled": self.temp_settings.sound_enabled,
                "ai_pause_time": self.temp_settings.ai_pause_time,
                "player_1_name": self.player_1_name_var.get(),
                "player_2_name": self.player_2_name_var.get(),
                "al1_name": self.al1_name_var.get(),
                "al2_name": self.al2_name_var.get(),
                "player_1_color": self.player_1_color_var.get(),
                "player_2_color": self.player_2_color_var.get(),
                "board_color_1": self.board_color_1_var.get(),
                "board_color_2": self.board_color_2_var.get(),
                "player_1_image": self.temp_settings.player_1_image,
                "player_2_image": self.temp_settings.player_2_image,
                "al1_image": self.temp_settings.al1_image,
                "al2_image": self.temp_settings.al2_image,
                "player_1_piece_image": self.temp_settings.player_1_piece_image,
                "player_1_king_image": self.temp_settings.player_1_king_image,
                "player_2_piece_image": self.temp_settings.player_2_piece_image,
                "player_2_king_image": self.temp_settings.player_2_king_image,
                "player_1_ai_type": self.player_1_ai_type_var.get(),
                "player_2_ai_type": self.player_2_ai_type_var.get()
            })
            save_config(config)

            # به‌روزرسانی ai_config با تنظیمات پیش‌فرض
            ai_config["ai_configs"] = self.temp_settings.ai_configs.copy()
            default_params = {
                "ability_level": 5,
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
            }
            for player in ["player_1", "player_2"]:
                ai_type = ai_config["ai_configs"][player]["ai_type"]
                if ai_type != "none":
                    ai_config["ai_configs"][player]["ability_level"] = default_params["ability_level"]
                    ai_config["ai_configs"][player]["training_params"] = default_params["training_params"].copy()
                    ai_config["ai_configs"][player]["reward_weights"] = default_params["reward_weights"].copy()
                else:
                    ai_config["ai_configs"][player]["ability_level"] = default_params["ability_level"]
                    ai_config["ai_configs"][player]["training_params"] = {}
                    ai_config["ai_configs"][player]["reward_weights"] = {}
            save_ai_config(ai_config)

            if config.get("language") != self.interface.settings.language:
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
        except json.JSONDecodeError as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"فرمت JSON نامعتبر: {str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در ذخیره تنظیمات: {str(e)}"
            )

    def reset(self):
        """بازنشانی تنظیمات به مقادیر پیش‌فرض و ذخیره در فایل‌های JSON."""
        print("Reset called")  # لاگ دیباگ
        try:
            config = load_config()
            print("Config loaded:", config)  # لاگ دیباگ
            # ایجاد نمونه جدید از temp_settings
            self.temp_settings = type(self.temp_settings)()
            # تنظیمات عمومی
            self.temp_settings.language = config.get("language", "en")
            self.temp_settings.game_mode = config.get("game_mode", "human_vs_human")
            self.temp_settings.ai_vs_ai_mode = config.get("ai_vs_ai_mode", "only_once")
            self.temp_settings.repeat_hands = config.get("repeat_hands", 1)
            self.temp_settings.player_starts = config.get("player_starts", True)
            self.temp_settings.use_timer = config.get("use_timer", False)
            self.temp_settings.game_time = config.get("game_time", 5)
            self.temp_settings.piece_style = config.get("piece_style", "classic")
            self.temp_settings.sound_enabled = config.get("sound_enabled", True)
            self.temp_settings.ai_pause_time = config.get("ai_pause_time", 100)
            self.temp_settings.player_1_name = config.get("player_1_name", "Player 1")
            self.temp_settings.player_2_name = config.get("player_2_name", "Player 2")
            self.temp_settings.al1_name = config.get("al1_name", "AI 1")
            self.temp_settings.al2_name = config.get("al2_name", "AI 2")
            # تنظیم رنگ‌ها
            self.temp_settings.player_1_color = hex_to_rgb(config.get("player_1_color", "#ff0000"))
            self.temp_settings.player_2_color = hex_to_rgb(config.get("player_2_color", "#0000ff"))
            self.temp_settings.board_color_1 = hex_to_rgb(config.get("board_color_1", "#ffffff"))
            self.temp_settings.board_color_2 = hex_to_rgb(config.get("board_color_2", "#8b4513"))
            # تنظیمات AI
            self.temp_settings.ai_configs = {
                "player_1": {
                    "ai_type": "none",
                    "ability_level": 5,
                    "training_params": config.get("default_ai_params", {}).get("training_params", {}),
                    "reward_weights": config.get("default_ai_params", {}).get("reward_weights", {})
                },
                "player_2": {
                    "ai_type": "none",
                    "ability_level": 5,
                    "training_params": config.get("default_ai_params", {}).get("training_params", {}),
                    "reward_weights": config.get("default_ai_params", {}).get("reward_weights", {})
                }
            }
            print("Temp settings after reset:", vars(self.temp_settings))  # لاگ دیباگ
            self.update_ui()
            self.save()  # ذخیره تنظیمات پیش‌فرض در فایل‌های JSON
            print("Reset completed, settings saved")  # لاگ دیباگ
            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_reset"]
            )
        except Exception as e:
            print(f"Error in reset: {str(e)}")  # لاگ دیباگ
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
        if hasattr(self, 'sound_var'):
            self.sound_var.set(self.temp_settings.sound_enabled)
        if hasattr(self, 'ai_pause_var'):
            self.ai_pause_var.set(self.temp_settings.ai_pause_time)
        self.player_1_name_var.set(self.temp_settings.player_1_name)
        self.player_2_name_var.set(self.temp_settings.player_2_name)
        self.al1_name_var.set(self.temp_settings.al1_name)
        self.al2_name_var.set(self.temp_settings.al2_name)
        # تنظیم رنگ‌ها با فرمت هگز
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
        if hasattr(self, 'player_1_ability_var'):
            self.player_1_ability_var.set(
                LANGUAGES[self.settings.language][
                    ability_mapping.get(self.temp_settings.ai_configs["player_1"]["ability_level"], "medium")
                ]
            )
        if hasattr(self, 'player_2_ability_var'):
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

class ALProgressWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["al_progress"],
                         self.config.get("progress_window_width", 600),
                         self.config.get("progress_window_height", 400))

        table_frame = ttk.Frame(self.window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        headers = ["Parameter", "Shape", "Num Elements"]
        if self.settings.language == "fa":
            headers = ["پارامتر", "شکل", "تعداد عناصر"]

        for col, header in enumerate(headers):
            ttk.Label(table_frame, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col, padx=5, pady=2, sticky="w")

        model_data = {}
        total_params = 0
        pth_file = "1/al_model.pth"

        if os.path.exists(pth_file):
            try:
                model_data = torch.load(pth_file, map_location=torch.device('cpu'))
                for key, tensor in model_data.items():
                    total_params += tensor.numel()
            except Exception as e:
                model_data = {"Error": f"Failed to load model: {e}"}
        else:
            model_data = {"Error": "Model file not found."}

        if "Error" not in model_data:
            for row, (key, tensor) in enumerate(model_data.items(), 1):
                ttk.Label(table_frame, text=key).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(table_frame, text=str(list(tensor.shape))).grid(row=row, column=1, sticky="w", padx=5, pady=2)
                ttk.Label(table_frame, text=str(tensor.numel())).grid(row=row, column=2, sticky="w", padx=5, pady=2)

            ttk.Label(table_frame, text="Total Parameters" if self.settings.language == "en" else "مجموع پارامترها",
                     font=("Arial", 10, "bold")).grid(row=row + 1, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(table_frame, text=str(total_params)).grid(row=row + 1, column=2, sticky="w", padx=5, pady=2)
        else:
            ttk.Label(table_frame, text=model_data["Error"]).grid(row=1, column=0, columnspan=3, padx=5, pady=10)

        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(pady=10)

class AZProgressWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["az_progress"],
                         self.config.get("progress_window_width", 600),
                         self.config.get("progress_window_height", 400))

        table_frame = ttk.Frame(self.window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        headers = ["Parameter", "Shape", "Num Elements"]
        if self.settings.language == "fa":
            headers = ["پارامتر", "شکل", "تعداد عناصر"]

        for col, header in enumerate(headers):
            ttk.Label(table_frame, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col, padx=5, pady=2, sticky="w")

        model_data = {}
        total_params = 0
        pth_file = "1/az_model.pth"

        if os.path.exists(pth_file):
            try:
                model_data = torch.load(pth_file, map_location=torch.device('cpu'))
                for key, tensor in model_data.items():
                    total_params += tensor.numel()
            except Exception as e:
                model_data = {"Error": f"Failed to load model: {e}"}
        else:
            model_data = {"Error": "Model file not found."}

        if "Error" not in model_data:
            for row, (key, tensor) in enumerate(model_data.items(), 1):
                ttk.Label(table_frame, text=key).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                ttk.Label(table_frame, text=str(list(tensor.shape))).grid(row=row, column=1, sticky="w", padx=5, pady=2)
                ttk.Label(table_frame, text=str(tensor.numel())).grid(row=row, column=2, sticky="w", padx=5, pady=2)

            ttk.Label(table_frame, text="Total Parameters" if self.settings.language == "en" else "مجموع پارامترها",
                     font=("Arial", 10, "bold")).grid(row=row + 1, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(table_frame, text=str(total_params)).grid(row=row + 1, column=2, sticky="w", padx=5, pady=2)
        else:
            ttk.Label(table_frame, text=model_data["Error"]).grid(row=1, column=0, columnspan=3, padx=5, pady=10)

        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"],
                   command=self.close, style="Custom.TButton").pack(pady=10)

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
        self.title(LANGUAGES[language]["advanced_settings_title"])
        self.geometry("600x500")
        self.resizable(False, False)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.setup_player_tab(notebook, "player_1", LANGUAGES[language]["player_1"])
        self.setup_player_tab(notebook, "player_2", LANGUAGES[language]["player_2"])

        ttk.Button(self, text=LANGUAGES[language]["save_all"],
                   command=self.save_advanced_settings,
                   style="Custom.TButton").pack(pady=10)

    def setup_player_tab(self, notebook, player_key, player_name):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text=player_name)

        training_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.settings.language]["training_params"], padding=10)
        training_frame.pack(fill="x", pady=5)

        training_params = self.temp_settings.ai_configs.get(player_key, {}).get("training_params", {})

        learning_rate_var = tk.DoubleVar(value=training_params.get("learning_rate", 0.001))
        ttk.Label(training_frame, text="Learning Rate:").grid(row=0, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=learning_rate_var).grid(row=0, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_learning_rate_var", learning_rate_var)

        gamma_var = tk.DoubleVar(value=training_params.get("gamma", 0.99))
        ttk.Label(training_frame, text="Gamma:").grid(row=1, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=gamma_var).grid(row=1, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_gamma_var", gamma_var)

        batch_size_var = tk.IntVar(value=training_params.get("batch_size", 128))
        ttk.Label(training_frame, text="Batch Size:").grid(row=2, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=batch_size_var).grid(row=2, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_batch_size_var", batch_size_var)

        memory_size_var = tk.IntVar(value=training_params.get("memory_size", 10000))
        ttk.Label(training_frame, text="Memory Size:").grid(row=3, column=0, padx=5, sticky="e")
        ttk.Entry(training_frame, textvariable=memory_size_var).grid(row=3, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_memory_size_var", memory_size_var)

        reward_frame = ttk.LabelFrame(frame, text=LANGUAGES[self.settings.language]["reward_weights"], padding=10)
        reward_frame.pack(fill="x", pady=5)

        reward_weights = self.temp_settings.ai_configs.get(player_key, {}).get("reward_weights", {})

        piece_difference_var = tk.DoubleVar(value=reward_weights.get("piece_difference", 1.0))
        ttk.Label(reward_frame, text="Piece Difference:").grid(row=0, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=piece_difference_var).grid(row=0, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_piece_difference_var", piece_difference_var)

        king_bonus_var = tk.DoubleVar(value=reward_weights.get("king_bonus", 2.0))
        ttk.Label(reward_frame, text="King Bonus:").grid(row=1, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=king_bonus_var).grid(row=1, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_king_bonus_var", king_bonus_var)

        position_bonus_var = tk.DoubleVar(value=reward_weights.get("position_bonus", 0.1))
        ttk.Label(reward_frame, text="Position Bonus:").grid(row=2, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=position_bonus_var).grid(row=2, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_position_bonus_var", position_bonus_var)

        capture_bonus_var = tk.DoubleVar(value=reward_weights.get("capture_bonus", 1.0))
        ttk.Label(reward_frame, text="Capture Bonus:").grid(row=3, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=capture_bonus_var).grid(row=3, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_capture_bonus_var", capture_bonus_var)

        multi_jump_bonus_var = tk.DoubleVar(value=reward_weights.get("multi_jump_bonus", 2.0))
        ttk.Label(reward_frame, text="Multi-Jump Bonus:").grid(row=4, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=multi_jump_bonus_var).grid(row=4, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_multi_jump_bonus_var", multi_jump_bonus_var)

        king_capture_bonus_var = tk.DoubleVar(value=reward_weights.get("king_capture_bonus", 3.0))
        ttk.Label(reward_frame, text="King Capture Bonus:").grid(row=5, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=king_capture_bonus_var).grid(row=5, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_king_capture_bonus_var", king_capture_bonus_var)

        mobility_bonus_var = tk.DoubleVar(value=reward_weights.get("mobility_bonus", 0.1))
        ttk.Label(reward_frame, text="Mobility Bonus:").grid(row=6, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=mobility_bonus_var).grid(row=6, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_mobility_bonus_var", mobility_bonus_var)

        safety_penalty_var = tk.DoubleVar(value=reward_weights.get("safety_penalty", -0.5))
        ttk.Label(reward_frame, text="Safety Penalty:").grid(row=7, column=0, padx=5, sticky="e")
        ttk.Entry(reward_frame, textvariable=safety_penalty_var).grid(row=7, column=1, padx=5, sticky="ew")
        setattr(self, f"{player_key}_safety_penalty_var", safety_penalty_var)

    def save_advanced_settings(self):
        try:
            for player in ["player_1", "player_2"]:
                learning_rate = getattr(self, f"{player}_learning_rate_var").get()
                gamma = getattr(self, f"{player}_gamma_var").get()
                batch_size = getattr(self, f"{player}_batch_size_var").get()
                memory_size = getattr(self, f"{player}_memory_size_var").get()

                if not (0 < learning_rate <= 1.0):
                    raise ValueError("Learning Rate باید بین 0 و 1 باشد")
                if not (0 <= gamma <= 1.0):
                    raise ValueError("Gamma باید بین 0 و 1 باشد")
                if batch_size <= 0:
                    raise ValueError("Batch Size باید مثبت باشد")
                if memory_size <= 0:
                    raise ValueError("Memory Size باید مثبت باشد")

            player1_training_params = {
                "learning_rate": self.player_1_learning_rate_var.get(),
                "gamma": self.player_1_gamma_var.get(),
                "batch_size": self.player_1_batch_size_var.get(),
                "memory_size": self.player_1_memory_size_var.get()
            }
            player1_reward_weights = {
                "piece_difference": self.player_1_piece_difference_var.get(),
                "king_bonus": self.player_1_king_bonus_var.get(),
                "position_bonus": self.player_1_position_bonus_var.get(),
                "capture_bonus": self.player_1_capture_bonus_var.get(),
                "multi_jump_bonus": self.player_1_multi_jump_bonus_var.get(),
                "king_capture_bonus": self.player_1_king_capture_bonus_var.get(),
                "mobility_bonus": self.player_1_mobility_bonus_var.get(),
                "safety_penalty": self.player_1_safety_penalty_var.get()
            }
            self.temp_settings.ai_configs["player_1"]["training_params"] = player1_training_params
            self.temp_settings.ai_configs["player_1"]["reward_weights"] = player1_reward_weights

            player2_training_params = {
                "learning_rate": self.player_2_learning_rate_var.get(),
                "gamma": self.player_2_gamma_var.get(),
                "batch_size": self.player_2_batch_size_var.get(),
                "memory_size": self.player_2_memory_size_var.get()
            }
            player2_reward_weights = {
                "piece_difference": self.player_2_piece_difference_var.get(),
                "king_bonus": self.player_2_king_bonus_var.get(),
                "position_bonus": self.player_2_position_bonus_var.get(),
                "capture_bonus": self.player_2_capture_bonus_var.get(),
                "multi_jump_bonus": self.player_2_multi_jump_bonus_var.get(),
                "king_capture_bonus": self.player_2_king_capture_bonus_var.get(),
                "mobility_bonus": self.player_2_mobility_bonus_var.get(),
                "safety_penalty": self.player_2_safety_penalty_var.get()
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
            self.destroy()  # بستن پنجره تنظیمات پیشرفته پس از ذخیره
        except Exception as e:
            messagebox.showerror(
                LANGUAGES[self.settings.language]["error"],
                f"خطا در ذخیره تنظیمات پیشرفته: {str(e)}"
            )