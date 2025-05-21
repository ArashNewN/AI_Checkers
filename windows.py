import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
from .game import Game
from .settings import GameSettings
from .config import save_config, load_config, LANGUAGES
from .utils import hex_to_rgb, rgb_to_hex
import torch
import os

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
        configure_styles()

    def create_window(self, title, width=400, height=300):
        if self.is_open or self.window:
            return
        self.window = tk.Toplevel(self.root) if self.root else tk.Toplevel()
        self.window.title(title)
        config = load_config()
        x = (config['window_width'] - width) // 2
        y = (config['window_height'] - height) // 2
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        self.window.minsize(300, 200)
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

class GameWindow(BaseWindow):
    def __init__(self, interface):
        super().__init__(interface)
        self.config = load_config()
        self.root = tk.Tk()
        self.root.title(f"Checkers Game - Version {self.config['game_version']}")
        self.root.geometry(f"{self.config['window_width']}x{self.config['window_height']}")
        self.root.minsize(600, 400)
        self.canvas = None
        self.settings_window = None
        self.create_widgets()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=self.config['window_width'], height=self.config['window_height'])
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.interface.handle_click)
        self.interface.draw_board(self.canvas)

        settings_button = ttk.Button(
            self.root,
            text=LANGUAGES[self.interface.settings.language]["settings"],
            command=self.open_settings,
            style="Custom.TButton"
        )
        settings_button.place(x=10, y=10)

    def open_settings(self):
        if not self.settings_window or not self.settings_window.is_open:
            self.settings_window = SettingsWindow(self.interface, self.root)
            self.settings_window.create_widgets()

    def destroy(self):
        self.root.destroy()

class SettingsWindow(BaseWindow):
    def __init__(self, interface, root=None):
        super().__init__(interface, root)
        self.config = load_config()
        self.temp_settings = GameSettings()
        self.copy_current_settings()

    def copy_current_settings(self):
        loaded_config = load_config()
        for key, value in loaded_config.items():
            if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                value = hex_to_rgb(value)
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
            if hasattr(self.temp_settings, key):
                setattr(self.temp_settings, key, value)

    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["settings"], 500, 750)
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
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["save_changes"], command=self.save,
                   style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["close"], command=self.close,
                   style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.settings.language]["reset_settings"], command=self.reset,
                   style="Custom.TButton").pack(side=side, padx=5)

        if self.play_with_var.get() == "ai_vs_ai":
            self.repeat_once_rb.config(state="normal")
            self.repeat_until_rb.config(state="normal")
        self.window.protocol("WM_DELETE_WINDOW", self.close)

    def setup_game_tab(self, notebook):
        game_frame = ttk.Frame(notebook, padding="10")
        notebook.add(game_frame, text=LANGUAGES[self.settings.language]["game_settings_tab"])

        lang_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["language"], padding=10)
        lang_frame.pack(fill="x", pady=5)
        self.lang_var = tk.StringVar(value=self.temp_settings.language)
        lang_menu = ttk.OptionMenu(lang_frame, self.lang_var, self.temp_settings.language, "en", "fa")
        lang_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        lang_frame.grid_columnconfigure(0, weight=1)
        lang_menu["menu"].config(font=("Arial", 10))

        play_with_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["play_with"], padding=10)
        play_with_frame.pack(fill="x", pady=5)
        self.play_with_var = tk.StringVar(value=self.temp_settings.game_mode)
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["human_vs_human"], variable=self.play_with_var,
                        value="human_vs_human", command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["human_vs_ai"], variable=self.play_with_var,
                        value="human_vs_ai", command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)
        ttk.Radiobutton(play_with_frame, text=LANGUAGES[self.settings.language]["ai_vs_ai"], variable=self.play_with_var,
                        value="ai_vs_ai", command=self.toggle_ai_vs_ai_options).pack(side="left", padx=10)

        self.ai_vs_ai_subframe = ttk.Frame(game_frame, padding=10)
        self.ai_vs_ai_subframe.pack(fill="x", pady=5)
        self.ai_vs_ai_var = tk.StringVar(value=self.temp_settings.ai_vs_ai_mode)
        self.repeat_once_rb = ttk.Radiobutton(self.ai_vs_ai_subframe, text=LANGUAGES[self.settings.language]["only_once"],
                                              variable=self.ai_vs_ai_var, value="only_once",
                                              command=self.toggle_repeat_options, state="disabled")
        self.repeat_once_rb.pack(side="left", padx=10)
        self.repeat_until_rb = ttk.Radiobutton(self.ai_vs_ai_subframe, text=LANGUAGES[self.settings.language]["repeat_game"],
                                               variable=self.ai_vs_ai_var, value="repeat_game",
                                               command=self.toggle_repeat_options, state="disabled")
        self.repeat_until_rb.pack(side="left", padx=10)

        self.repeat_hands_frame = ttk.Frame(self.ai_vs_ai_subframe)
        self.hmh_var = tk.IntVar(value=self.temp_settings.repeat_hands)
        self.hmh_entry = ttk.Entry(self.repeat_hands_frame, textvariable=self.hmh_var, width=5)
        self.hmh_entry.pack(side="left", padx=5)
        self.toggle_repeat_options()

        start_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["starting_player"], padding=10)
        start_frame.pack(fill="x", pady=5)
        self.start_var = tk.StringVar(value="player_1" if self.temp_settings.player_starts else "player_2")
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["player"], variable=self.start_var,
                        value="player_1").pack(side="left", padx=10)
        ttk.Radiobutton(start_frame, text=LANGUAGES[self.settings.language]["ai"], variable=self.start_var,
                        value="player_2").pack(side="left", padx=10)

        timer_frame = ttk.LabelFrame(game_frame, text=LANGUAGES[self.settings.language]["game_timer"], padding=10)
        timer_frame.pack(fill="x", pady=5)
        self.timer_var = tk.StringVar(value="with_timer" if self.temp_settings.use_timer else "no_timer")
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["no_timer"], variable=self.timer_var,
                        value="no_timer", command=self.toggle_timer).pack(side="left", padx=8)
        ttk.Radiobutton(timer_frame, text=LANGUAGES[self.settings.language]["with_timer"], variable=self.timer_var,
                        value="with_timer", command=self.toggle_timer).pack(side="left", padx=8)
        self.timer_duration_frame = ttk.Frame(timer_frame)
        self.timer_var_duration = tk.IntVar(value=self.temp_settings.game_time)
        ttk.Label(self.timer_duration_frame, text=LANGUAGES[self.settings.language]["game_duration"]).pack(side="left")
        self.timer_combo = ttk.Combobox(self.timer_duration_frame, textvariable=self.timer_var_duration,
                                        state="readonly", values=[1, 2, 3, 5, 10, 20])
        self.timer_combo.pack(side="left", padx=5)
        if self.temp_settings.use_timer:
            self.timer_duration_frame.pack(side="left", padx=5)

    def setup_design_tab(self, notebook):
        design_frame = ttk.Frame(notebook, padding="10")
        notebook.add(design_frame, text=LANGUAGES[self.settings.language]["design_tab"])

        side = "right" if self.settings.language == "fa" else "left"
        anchor = "e" if side == "left" else "w"

        style_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["piece_style"], padding=10)
        style_frame.pack(fill="x", pady=5)
        self.piece_style_var = tk.StringVar(value=self.temp_settings.piece_style)
        style_menu = ttk.OptionMenu(style_frame, self.piece_style_var, self.temp_settings.piece_style, "circle", "outlinedocentric_circle", "square")
        style_menu.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        style_frame.grid_columnconfigure(0, weight=1)
        style_menu["menu"].config(font=("Arial", 10))

        color_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["color_settings"], padding=10)
        color_frame.pack(fill="x", pady=5)

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["player_piece_color"]).grid(row=0, column=0, padx=5, pady=5, sticky=anchor)
        self.player_1_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_1_color))
        self.p1_color_button = ttk.Button(color_frame, text="    ", command=lambda: self.choose_color(self.player_1_color_var, self.p1_color_button), style="Custom.TButton")
        self.p1_color_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["ai_piece_color"]).grid(row=1, column=0, padx=5, pady=5, sticky=anchor)
        self.player_2_color_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.player_2_color))
        self.p2_color_button = ttk.Button(color_frame, text="    ", command=lambda: self.choose_color(self.player_2_color_var, self.p2_color_button), style="Custom.TButton")
        self.p2_color_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["board_color_1"]).grid(row=2, column=0, padx=5, pady=5, sticky=anchor)
        self.board_color_1_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_1))
        self.b1_color_button = ttk.Button(color_frame, text="    ", command=lambda: self.choose_color(self.board_color_1_var, self.b1_color_button), style="Custom.TButton")
        self.b1_color_button.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())

        ttk.Label(color_frame, text=LANGUAGES[self.settings.language]["board_color_2"]).grid(row=3, column=0, padx=5, pady=5, sticky=anchor)
        self.board_color_2_var = tk.StringVar(value=rgb_to_hex(self.temp_settings.board_color_2))
        self.b2_color_button = ttk.Button(color_frame, text="    ", command=lambda: self.choose_color(self.board_color_2_var, self.b2_color_button), style="Custom.TButton")
        self.b2_color_button.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())

        piece_image_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["piece_images"], padding=10)
        piece_image_frame.pack(fill="x", pady=5)

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_1_piece"]).grid(row=0, column=0, padx=5, pady=5, sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_piece_image("player_1_piece"), style="Custom.TButton").grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_1_king"]).grid(row=1, column=0, padx=5, pady=5, sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_piece_image("player_1_king"), style="Custom.TButton").grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_2_piece"]).grid(row=2, column=0, padx=5, pady=5, sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_piece_image("player_2_piece"), style="Custom.TButton").grid(row=2, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(piece_image_frame, text=LANGUAGES[self.settings.language]["player_2_king"]).grid(row=3, column=0, padx=5, pady=5, sticky=anchor)
        ttk.Button(piece_image_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_piece_image("player_2_king"), style="Custom.TButton").grid(row=3, column=1, padx=5, pady=5, sticky="w")

        sound_frame = ttk.LabelFrame(design_frame, text=LANGUAGES[self.settings.language]["sound_settings"], padding=10)
        sound_frame.pack(fill="x", pady=5)
        self.sound_var = tk.BooleanVar(value=self.temp_settings.sound_enabled)
        ttk.Radiobutton(sound_frame, text=LANGUAGES[self.settings.language]["sound_on"], variable=self.sound_var, value=True).pack(side=side, padx=10)
        ttk.Radiobutton(sound_frame, text=LANGUAGES[self.settings.language]["sound_off"], variable=self.sound_var, value=False).pack(side=side, padx=10)

    def setup_ai_tab(self, notebook):
        ai_frame = ttk.Frame(notebook, padding="10")
        notebook.add(ai_frame, text=LANGUAGES[self.settings.language]["ai_tab"])

        side = "right" if self.settings.language == "fa" else "left"

        ttk.Label(ai_frame, text=LANGUAGES[self.settings.language]["player_1_ai_type"]).pack(anchor="w" if side == "left" else "e")
        self.player_1_ai_type_var = tk.StringVar(value=self.temp_settings.player_1_ai_type)
        player_1_ai_menu = ttk.OptionMenu(ai_frame, self.player_1_ai_type_var, self.temp_settings.player_1_ai_type, "none", "advanced", "alphazero")
        player_1_ai_menu.pack(fill="x", pady=5)
        player_1_ai_menu["menu"].config(font=("Arial", 10))

        ttk.Label(ai_frame, text=LANGUAGES[self.settings.language]["player_2_ai_type"]).pack(anchor="w" if side == "left" else "e")
        self.player_2_ai_type_var = tk.StringVar(value=self.temp_settings.player_2_ai_type)
        player_2_ai_menu = ttk.OptionMenu(ai_frame, self.player_2_ai_type_var, self.temp_settings.player_2_ai_type, "none", "advanced", "alphazero")
        player_2_ai_menu.pack(fill="x", pady=5)
        player_2_ai_menu["menu"].config(font=("Arial", 10))

        ttk.Label(ai_frame, text=LANGUAGES[self.settings.language]["ai_pause_time"]).pack(anchor="w" if side == "left" else "e")
        self.ai_pause_var = tk.IntVar(value=self.temp_settings.ai_pause_time)
        ttk.Entry(ai_frame, textvariable=self.ai_pause_var, width=10).pack(fill="x", pady=5)
        ttk.Label(ai_frame, text=LANGUAGES[self.settings.language]["ms"]).pack(anchor="w" if side == "left" else "e")

    def setup_player_tab(self, notebook):
        player_frame = ttk.Frame(notebook, padding="10")
        notebook.add(player_frame, text=LANGUAGES[self.settings.language]["player_tab"])

        side = "right" if self.settings.language == "fa" else "left"
        anchor = "e" if side == "left" else "w"

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_1_name"]).grid(row=0, column=0, padx=5, pady=5, sticky=anchor)
        self.player_1_name_var = tk.StringVar(value=self.temp_settings.player_1_name)
        ttk.Entry(player_frame, textvariable=self.player_1_name_var, width=15).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_image("player_1"), style="Custom.TButton").grid(row=0, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["player_2_name"]).grid(row=1, column=0, padx=5, pady=5, sticky=anchor)
        self.player_2_name_var = tk.StringVar(value=self.temp_settings.player_2_name)
        ttk.Entry(player_frame, textvariable=self.player_2_name_var, width=15).grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_image("player_2"), style="Custom.TButton").grid(row=1, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["ai_1_name"]).grid(row=2, column=0, padx=5, pady=5, sticky=anchor)
        self.ai_1_name_var = tk.StringVar(value=self.temp_settings.ai_1_name)
        ttk.Entry(player_frame, textvariable=self.ai_1_name_var, width=15).grid(row=2, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_image("ai_1"), style="Custom.TButton").grid(row=2, column=2, padx=5, pady=5, sticky="w")

        ttk.Label(player_frame, text=LANGUAGES[self.settings.language]["ai_2_name"]).grid(row=3, column=0, padx=5, pady=5, sticky=anchor)
        self.ai_2_name_var = tk.StringVar(value=self.temp_settings.ai_2_name)
        ttk.Entry(player_frame, textvariable=self.ai_2_name_var, width=15).grid(row=3, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(player_frame, text=LANGUAGES[self.settings.language]["upload_image"], command=lambda: self.upload_image("ai_2"), style="Custom.TButton").grid(row=3, column=2, padx=5, pady=5, sticky="w")

    def setup_advanced_tab(self, notebook):
        advanced_frame = ttk.Frame(notebook, padding="10")
        notebook.add(advanced_frame, text=LANGUAGES[self.settings.language]["advanced_settings_tab"])

        warning_label = ttk.Label(
            advanced_frame,
            text=LANGUAGES[self.settings.language]["advanced_settings_warning"],
            wraplength=350,
            justify="center",
            foreground="red"
        )
        warning_label.pack(pady=10)

        ability_frame = ttk.LabelFrame(advanced_frame, text=LANGUAGES[self.settings.language]["ai_ability_level"], padding=10)
        ability_frame.pack(fill="x", pady=5)
        self.ai_ability_level_var = tk.StringVar(value="medium")
        ability_levels = [
            ("very_weak", 1, LANGUAGES[self.settings.language]["very_weak"]),
            ("weak", 3, LANGUAGES[self.settings.language]["weak"]),
            ("medium", 5, LANGUAGES[self.settings.language]["medium"]),
            ("strong", 7, LANGUAGES[self.settings.language]["strong"]),
            ("very_strong", 9, LANGUAGES[self.settings.language]["very_strong"])
        ]
        ability_menu = ttk.Combobox(
            ability_frame,
            textvariable=self.ai_ability_level_var,
            values=[label for _, _, label in ability_levels],
            state="readonly"
        )
        ability_menu.pack(fill="x", pady=5)
        ability_menu.bind("<<ComboboxSelected>>", self.update_ability_levels)
        current_ability = self.temp_settings.ai_1_ability
        for level_key, value, label in ability_levels:
            if value == current_ability:
                self.ai_ability_level_var.set(label)
                break

        advanced_config_button = ttk.Button(
            advanced_frame,
            text=LANGUAGES[self.settings.language]["advanced_config"],
            command=self.open_advanced_config,
            style="Custom.TButton"
        )
        advanced_config_button.pack(pady=10)

    def update_ability_levels(self, event=None):
        ability_levels = {
            LANGUAGES[self.settings.language]["very_weak"]: 1,
            LANGUAGES[self.settings.language]["weak"]: 3,
            LANGUAGES[self.settings.language]["medium"]: 5,
            LANGUAGES[self.settings.language]["strong"]: 7,
            LANGUAGES[self.settings.language]["very_strong"]: 9
        }
        selected_level = self.ai_ability_level_var.get()
        ability_value = ability_levels.get(selected_level, 5)
        self.temp_settings.az1_ability = ability_value
        self.temp_settings.az2_ability = ability_value
        self.temp_settings.ai_1_ability = ability_value
        self.temp_settings.ai_2_ability = ability_value

    def open_advanced_config(self):
        if hasattr(self, "advanced_config_window") and self.advanced_config_window.is_open:
            return
        self.advanced_config_window = AdvancedConfigWindow(self, self.window)
        self.advanced_config_window.create_widgets()

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
        try:
            button.configure(style=f"Color_{id(button)}.TButton")
            style = ttk.Style()
            style.configure(f"Color_{id(button)}.TButton", background=color)
        except ValueError:
            pass

    def upload_image(self, player):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            if player == "player_1":
                self.temp_settings.player_1_image = file_path
            elif player == "player_2":
                self.temp_settings.player_2_image = file_path
            elif player == "ai_1":
                self.temp_settings.ai_1_image = file_path
            elif player == "ai_2":
                self.temp_settings.ai_2_image = file_path

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
        try:
            pause_time = self.ai_pause_var.get()
            if not 0 <= pause_time <= 5000:
                messagebox.showerror(LANGUAGES[self.settings.language]["error"], LANGUAGES[self.settings.language]["ai_pause_error"])
                return

            repeat_hands = self.hmh_var.get()
            if self.ai_vs_ai_var.get() == "repeat_game" and not 1 <= repeat_hands <= 1000:
                messagebox.showerror(LANGUAGES[self.settings.language]["error"], LANGUAGES[self.settings.language]["invalid_number_hands"])
                return

            config = {
                "player_starts": self.start_var.get() == "player_1",
                "use_timer": self.timer_var.get() == "with_timer",
                "game_time": self.timer_var_duration.get() if self.timer_var.get() == "with_timer" else self.temp_settings.game_time,
                "language": self.lang_var.get(),
                "player_1_color": self.player_1_color_var.get(),
                "player_2_color": self.player_2_color_var.get(),
                "board_color_1": self.board_color_1_var.get(),
                "board_color_2": self.board_color_2_var.get(),
                "piece_style": self.piece_style_var.get(),
                "sound_enabled": self.sound_var.get(),
                "ai_pause_time": pause_time,
                "az1_ability": self.temp_settings.az1_ability,
                "az2_ability": self.temp_settings.az2_ability,
                "ai_1_ability": self.temp_settings.ai_1_ability,
                "ai_2_ability": self.temp_settings.ai_2_ability,
                "game_mode": self.play_with_var.get(),
                "ai_vs_ai_mode": self.ai_vs_ai_var.get(),
                "repeat_hands": repeat_hands,
                "player_1_name": self.player_1_name_var.get() or "Player 1",
                "player_2_name": self.player_2_name_var.get() or "Player 2",
                "ai_1_name": self.ai_1_name_var.get() or "AI 1",
                "ai_2_name": self.ai_2_name_var.get() or "AI 2",
                "player_1_image": self.temp_settings.player_1_image or "",
                "player_2_image": self.temp_settings.player_2_image or "",
                "ai_1_image": self.temp_settings.ai_1_image or "",
                "ai_2_image": self.temp_settings.ai_2_image or "",
                "player_1_piece_image": self.temp_settings.player_1_piece_image or "",
                "player_1_king_image": self.temp_settings.player_1_king_image or "",
                "player_2_piece_image": self.temp_settings.player_2_piece_image or "",
                "player_2_king_image": self.temp_settings.player_2_king_image or "",
                "pause_between_hands": self.temp_settings.pause_between_hands,
                "player_1_ai_type": self.player_1_ai_type_var.get(),
                "player_2_ai_type": self.player_2_ai_type_var.get(),
            }

            save_config(config)

            for key, value in config.items():
                if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                    value = hex_to_rgb(value)
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

            for key, value in config.items():
                if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                    value = hex_to_rgb(value)
                if hasattr(self.temp_settings, key):
                    setattr(self.temp_settings, key, value)

            self.update_ui()

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

    def reset(self):
        if messagebox.askyesno(LANGUAGES[self.settings.language]["warning"], LANGUAGES[self.settings.language]["reset_settings_warning"]):
            default_config = load_config()
            self.temp_settings = GameSettings()
            self.play_with_var.set(default_config["game_mode"])
            self.ai_vs_ai_var.set(default_config["ai_vs_ai_mode"])
            self.hmh_var.set(default_config["repeat_hands"])
            self.start_var.set("player_1" if default_config["player_starts"] else "player_2")
            self.timer_var.set("with_timer" if default_config["use_timer"] else "no_timer")
            self.timer_var_duration.set(default_config["game_time"])
            self.lang_var.set(default_config["language"])
            self.player_1_color_var.set(default_config["player_1_color"])
            self.player_2_color_var.set(default_config["player_2_color"])
            self.board_color_1_var.set(default_config["board_color_1"])
            self.board_color_2_var.set(default_config["board_color_2"])
            self.piece_style_var.set(default_config["piece_style"])
            self.sound_var.set(default_config["sound_enabled"])
            self.ai_pause_var.set(default_config["ai_pause_time"])
            self.player_1_name_var.set(default_config["player_1_name"])
            self.player_2_name_var.set(default_config["player_1_name"])
            self.ai_1_name_var.set(default_config["ai_1_name"])
            self.ai_2_name_var.set(default_config["ai_2_name"])
            self.player_1_ai_type_var.set(default_config["player_1_ai_type"])
            self.player_2_ai_type_var.set(default_config["player_2_ai_type"])
            self.temp_settings.player_1_image = default_config["player_1_image"]
            self.temp_settings.player_2_image = default_config["player_2_image"]
            self.temp_settings.ai_1_image = default_config["ai_1_image"]
            self.temp_settings.ai_2_image = default_config["ai_2_image"]
            self.temp_settings.player_1_piece_image = default_config["player_1_piece_image"]
            self.temp_settings.player_1_king_image = default_config["player_1_king_image"]
            self.temp_settings.player_2_piece_image = default_config["player_2_piece_image"]
            self.temp_settings.player_2_king_image = default_config["player_2_king_image"]
            self.temp_settings.az1_ability = default_config["az1_ability"]
            self.temp_settings.az2_ability = default_config["az2_ability"]
            self.temp_settings.ai_1_ability = default_config["ai_1_ability"]
            self.temp_settings.ai_2_ability = default_config["ai_2_ability"]
            self.ai_ability_level_var.set(LANGUAGES[self.settings.language]["medium"])
            self.update_color_button(self.p1_color_button, self.player_1_color_var.get())
            self.update_color_button(self.p2_color_button, self.player_2_color_var.get())
            self.update_color_button(self.b1_color_button, self.board_color_1_var.get())
            self.update_color_button(self.b2_color_button, self.board_color_2_var.get())
            self.timer_combo.set(default_config["game_time"])
            self.toggle_ai_vs_ai_options()
            self.toggle_repeat_options()
            self.toggle_timer()
            self.save()

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
        self.ai_1_name_var.set(self.temp_settings.ai_1_name)
        self.ai_2_name_var.set(self.temp_settings.ai_2_name)
        self.player_1_color_var.set(rgb_to_hex(self.temp_settings.player_1_color))
        self.player_2_color_var.set(rgb_to_hex(self.temp_settings.player_2_color))
        self.board_color_1_var.set(rgb_to_hex(self.temp_settings.board_color_1))
        self.board_color_2_var.set(rgb_to_hex(self.temp_settings.board_color_2))
        self.player_1_ai_type_var.set(self.temp_settings.player_1_ai_type)
        self.player_2_ai_type_var.set(self.temp_settings.player_2_ai_type)
        self.update_color_button(self.p1_color_button, self.player_1_color_var.get())
        self.update_color_button(self.p2_color_button, self.player_2_color_var.get())
        self.update_color_button(self.b1_color_button, self.board_color_1_var.get())
        self.update_color_button(self.b2_color_button, self.board_color_2_var.get())
        self.toggle_ai_vs_ai_options()
        self.toggle_repeat_options()
        self.toggle_timer()

class AIProgressWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["ai_progress"], 600, 400)

        table_frame = ttk.Frame(self.window)
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        headers = ["Parameter", "Shape", "Num Elements"]
        if self.settings.language == "fa":
            headers = ["پارامتر", "شکل", "تعداد عناصر"]

        for col, header in enumerate(headers):
            ttk.Label(table_frame, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col, padx=5, pady=2, sticky="w")

        model_data = {}
        total_params = 0
        pth_file = "1/ai_model.pth"

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

            ttk.Label(table_frame, text="Total Parameters" if self.settings.language == "en" else "مجموع پارامترها", font=("Arial", 10, "bold")).grid(row=row + 1, column=0, sticky="w", padx=5, pady=2)
            ttk.Label(table_frame, text=str(total_params)).grid(row=row + 1, column=2, sticky="w", padx=5, pady=2)
        else:
            ttk.Label(table_frame, text=model_data["Error"]).grid(row=1, column=0, columnspan=3, padx=5, pady=10)

        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"], command=self.close, style="Custom.TButton").pack(pady=10)

class HelpWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["help"], 300, 200)
        ttk.Label(self.window, text="COMING SOON!", font=("Arial", 14)).pack(pady=50)
        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"], command=self.close, style="Custom.TButton").pack(pady=10)

class AboutWindow(BaseWindow):
    def create_widgets(self):
        self.create_window(LANGUAGES[self.settings.language]["about_me"], 300, 200)
        ttk.Label(self.window, text="COMING SOON!", font=("Arial", 14)).pack(pady=50)
        ttk.Button(self.window, text=LANGUAGES[self.settings.language]["close"], command=self.close, style="Custom.TButton").pack(pady=10)

class AdvancedConfigWindow(BaseWindow):
    def __init__(self, parent, root):
        super().__init__(parent.interface, root)
        self.parent = parent
        self.config = load_config()
        self.temp_config = self.config.copy()
        self.entries = {}

    def create_widgets(self):
        self.create_window(LANGUAGES[self.parent.settings.language]["advanced_config"], 500, 600)

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True)

        sections = [
            ("ai_1_training_params", LANGUAGES[self.parent.settings.language]["ai_1_training_params"]),
            ("ai_2_training_params", LANGUAGES[self.parent.settings.language]["ai_2_training_params"]),
            ("az1_training_params", LANGUAGES[self.parent.settings.language]["az1_training_params"]),
            ("az2_training_params", LANGUAGES[self.parent.settings.language]["az2_training_params"]),
            ("mcts_params", LANGUAGES[self.parent.settings.language]["mcts_params"]),
            ("network_params", LANGUAGES[self.parent.settings.language]["network_params"]),
            ("advanced_nn_params", LANGUAGES[self.parent.settings.language]["advanced_nn_params"])
        ]

        for section_key, section_label in sections:
            frame = ttk.Frame(notebook, padding="10")
            notebook.add(frame, text=section_label)
            self.create_section(frame, section_key)

        button_frame = ttk.Frame(self.window, padding=10)
        button_frame.pack(fill="x")
        side = "right" if self.parent.settings.language == "fa" else "left"
        ttk.Button(button_frame, text=LANGUAGES[self.parent.settings.language]["save_all"], command=self.save,
                   style="Custom.TButton").pack(side=side, padx=5)
        ttk.Button(button_frame, text=LANGUAGES[self.parent.settings.language]["close"], command=self.close,
                   style="Custom.TButton").pack(side=side, padx=5)

    def create_section(self, parent_frame, section_key):
        section_frame = ttk.LabelFrame(parent_frame, text=section_key, padding=10)
        section_frame.pack(fill="both", expand=True, pady=5)
        self.entries[section_key] = {}

        canvas = tk.Canvas(section_frame)
        scrollbar = ttk.Scrollbar(section_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for key, value in self.config[section_key].items():
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill="x", pady=2)
            ttk.Label(row_frame, text=key.replace("_", " ").title()).pack(side="left", padx=5)

            if isinstance(value, bool):
                var = tk.BooleanVar(value=value)
                ttk.Checkbutton(row_frame, variable=var).pack(side="left")
            else:
                var = tk.StringVar(value=str(value))
                ttk.Entry(row_frame, textvariable=var, width=20).pack(side="left", padx=5)

            self.entries[section_key][key] = var

        reset_button = ttk.Button(
            scrollable_frame,
            text=LANGUAGES[self.parent.settings.language]["reset_tab"],
            command=lambda k=section_key: self.reset_section(k),
            style="Custom.TButton"
        )
        reset_button.pack(pady=5)

    def reset_section(self, section_key):
        default_config = load_config()
        self.temp_config[section_key] = default_config[section_key].copy()
        for key, var in self.entries[section_key].items():
            if isinstance(var, tk.BooleanVar):
                var.set(default_config[section_key][key])
            else:
                var.set(str(default_config[section_key][key]))

    def save(self):
        try:
            for section_key, entries in self.entries.items():
                for key, var in entries.items():
                    value = var.get()
                    if isinstance(var, tk.BooleanVar):
                        self.temp_config[section_key][key] = value
                    else:
                        try:
                            if isinstance(self.config[section_key][key], int):
                                self.temp_config[section_key][key] = int(value)
                            elif isinstance(self.config[section_key][key], float):
                                self.temp_config[section_key][key] = float(value)
                            else:
                                self.temp_config[section_key][key] = value
                        except ValueError:
                            messagebox.showerror(LANGUAGES[self.settings.language]["error"],
                                                 LANGUAGES[self.settings.language]["invalid_input"])
                            return

            save_config(self.temp_config)
            self.parent.config = self.temp_config.copy()
            messagebox.showinfo(LANGUAGES[self.settings.language]["settings"],
                                LANGUAGES[self.settings.language]["apply_after_game"])
            self.close()

        except tk.TclError:
            self.close()