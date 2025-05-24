#Required classes from various modules to examine AI movement in the game

#windoows.py:
class SettingsWindow(BaseWindow):
    def __init__(self, interface, root=None):
        super().__init__(interface, root)
        self.temp_settings = GameSettings()
        self.copy_current_settings()
        self.ai_types = ["none"]
        self.entries = {}
        self.initialize_ai_modules()
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

    def initialize_ai_modules(self):
        try:
            from modules.base_ai import BaseAI
            config = _config_manager.load_config()
            ai_dirs = config.get("ai_module_dirs", ["modules"])
            ai_modules = []

            for ai_dir in ai_dirs:
                module_dir = self.project_dir / ai_dir
                module_dir.mkdir(parents=True, exist_ok=True)
                for py_file in module_dir.glob("*.py"):
                    module_name = py_file.stem
                    if module_name in ["__init__", "windows", "base_ai", "self_play", "train", "train_advanced"]:
                        logger.debug(f"Skipping file: {module_name}")
                        continue
                    module_path = None  # مقدار پیش‌فرض
                    try:
                        module_path = f"{ai_dir.replace('/', '.')}.{module_name}"
                        module = importlib.import_module(module_path)
                        logger.debug(f"Loaded module: {module_path}")
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, type) and issubclass(attr, BaseAI) and attr != BaseAI:
                                metadata = getattr(attr, "metadata", None) or attr.get_metadata()
                                if metadata and all(key in metadata for key in ["type", "description", "code"]):
                                    ai_modules.append({
                                        "display": module_name,
                                        "default_type": metadata["type"],
                                        "default_description": metadata["description"],
                                        "default_code": metadata["code"],
                                        "class_name": attr_name
                                    })
                                    logger.debug(f"Found AI: {module_name}.{attr_name}, metadata: {metadata}")
                                else:
                                    logger.debug(f"Skipped {module_name}.{attr_name}: Invalid metadata")
                    except Exception as e:
                        error_path = module_path if module_path else f"{ai_dir}/{module_name}"
                        logger.error(f"Error loading {error_path}: {str(e)}")
                        _config_manager.log_to_json(f"Error loading {error_path}: {str(e)}", level="ERROR")

            ai_config = _config_manager.load_ai_config()
            for module in ai_modules:
                ai_type = module["default_type"]
                code = module["default_code"]
                used_codes = {info.get("code") for info in ai_config["ai_types"].values()}
                if ai_type not in ai_config["ai_types"] and code not in used_codes:
                    ai_config["ai_types"][ai_type] = {
                        "module": f"modules.{module['display']}",
                        "class": module["class_name"],
                        "description": module["default_description"],
                        "code": code
                    }
                    _config_manager.save_ai_specific_config(code, {
                        "player_1": DEFAULT_AI_PARAMS.copy(),
                        "player_2": DEFAULT_AI_PARAMS.copy()
                    })
            _config_manager.save_ai_config(ai_config)
            self.ai_types = ["none"] + list(ai_config["ai_types"].keys())
            logger.debug(f"Initialized ai_types: {self.ai_types}")
            _config_manager.log_to_json("AI modules initialized", level="INFO", extra_data={"ai_types": self.ai_types})
        except Exception as e:
            logger.error(f"Error in initialize_ai_modules: {str(e)}")
            self.ai_types = ["none"]
            self.temp_settings.ai_configs = _config_manager.load_ai_config()
            _config_manager.log_to_json(f"Error in initialize_ai_modules: {str(e)}", level="ERROR")

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
        try:
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format, expected a dictionary")
            self.ai_types = ["none"] + list(ai_config.get("ai_types", {}).keys())
            self.temp_settings.ai_configs = ai_config.get("ai_configs", _config_manager.load_ai_config())
        except Exception as e:
            logger.error(f"Error loading ai_config: {e}, using default")
            self.ai_types = ["none"]
            self.temp_settings.ai_configs = _config_manager.load_ai_config()
            _config_manager.log_to_json(f"Error loading ai_config: {str(e)}", level="ERROR")
        logger.debug(f"Loaded ai_types: {self.ai_types}")

    @staticmethod
    def check_ai_module(ai_type: str) -> bool:
        if ai_type == "none":
            return True
        try:
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format")
            return ai_type in ai_config.get("ai_types", {})
        except Exception as e:
            logger.error(f"Error checking ai_module {ai_type}: {e}")
            return False

    def update_ai_dropdowns(self):
        try:
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format")
            logger.debug(f"Updating AI dropdowns with ai_types: {ai_config['ai_types'].keys()}")

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
                logger.info(f"Reset player_1 AI to 'none' (was {self.player_1_ai_type_var.get()})")
            if self.player_2_ai_type_var.get() not in self.ai_types:
                self.player_2_ai_type_var.set("none")
                self._update_player_ai_config("player_2", "none")
                logger.info(f"Reset player_2 AI to 'none' (was {self.player_2_ai_type_var.get()})")
        except Exception as e:
            logger.error(f"Error updating AI dropdowns: {e}")
            _config_manager.log_to_json(f"Error updating AI dropdowns: {str(e)}", level="ERROR")

    def _update_player_ai_config(self, player: str, ai_type: str):
        try:
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format")
            logger.debug(f"Updating {player} temp_settings to AI: {ai_type}")

            if ai_type != "none" and ai_type not in ai_config["ai_types"]:
                logger.error(f"AI type '{ai_type}' not found in ai_config['ai_types']")
                self.log_error(f"ماژول AI {ai_type} برای {player} پیدا نشد")
                self.player_1_ai_type_var.set("none") if player == "player_1" else self.player_2_ai_type_var.set("none")
                ai_type = "none"

            self.temp_settings.ai_configs[player]["ai_type"] = ai_type
            if ai_type != "none":
                code = ai_config["ai_types"][ai_type]["code"]
                self.temp_settings.ai_configs[player]["ai_code"] = code
                ai_specific_config = _config_manager.load_ai_specific_config(code)
                if not isinstance(ai_specific_config, dict):
                    logger.warning(f"Invalid ai_specific_config for {code}, using default")
                    ai_specific_config = {player: DEFAULT_AI_PARAMS.copy()}
                self.temp_settings.ai_configs[player]["params"] = ai_specific_config.get(player, DEFAULT_AI_PARAMS.copy())
                self.temp_settings.ai_configs[player]["ability_level"] = 5
            else:
                self.temp_settings.ai_configs[player]["ai_code"] = None
                self.temp_settings.ai_configs[player]["params"] = {}
                self.temp_settings.ai_configs[player]["ability_level"] = 5

            logger.debug(f"Updated {player} temp_settings: {self.temp_settings.ai_configs[player]}")
            _config_manager.log_to_json(f"Updated AI config for {player}", level="INFO", extra_data={"ai_type": ai_type})
        except Exception as e:
            logger.error(f"Error updating player AI config for {player}: {e}")
            _config_manager.log_to_json(f"Error updating player AI config for {player}: {str(e)}", level="ERROR")

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

    def update_ai_list(self):
        self.ai_list.delete(*self.ai_list.get_children())
        try:
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format")
            for ai_type, ai_info in ai_config["ai_types"].items():
                description = ai_info.get("description", "")
                code = ai_info.get("code", "")
                self.ai_list.insert("", "end", values=(ai_type, code, description))
        except Exception as e:
            logger.error(f"Error updating ai_list: {e}")
            _config_manager.log_to_json(f"Error updating ai_list: {str(e)}", level="ERROR")

    def copy_current_settings(self):
        try:
            config = _config_manager.load_config()
            if not isinstance(config, dict):
                raise ValueError("Invalid config format")
            ai_config = _config_manager.load_ai_config()
            if not isinstance(ai_config, dict):
                raise ValueError("Invalid ai_config format")
            for key, value in config.items():
                if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                    value = hex_to_rgb(value)
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
                if hasattr(self.temp_settings, key):
                    setattr(self.temp_settings, key, value)
            self.temp_settings.ai_configs = ai_config.get("ai_configs", _config_manager.load_ai_config())
        except Exception as e:
            logger.error(f"Error copying current settings: {e}")
            self.temp_settings.ai_configs = _config_manager.load_ai_config()
            _config_manager.log_to_json(f"Error copying current settings: {str(e)}", level="ERROR")

    def update_ability_levels(self, player: str):
        ability_levels = {
            LANGUAGES[self.settings.language]["very_weak"]: 1,
            LANGUAGES[self.settings.language]["weak"]: 3,
            LANGUAGES[self.settings.language]["medium"]: 5,
            LANGUAGES[self.settings.language]["strong"]: 7,
            LANGUAGES[self.settings.language]["very_strong"]: 9
        }
        try:
            if player == "player_1":
                selected_level = self.player_1_ability_var.get()
                self.temp_settings.ai_configs["player_1"]["ability_level"] = ability_levels.get(selected_level, 5)
            elif player == "player_2":
                selected_level = self.player_2_ability_var.get()
                self.temp_settings.ai_configs["player_2"]["ability_level"] = ability_levels.get(selected_level, 5)
            _config_manager.log_to_json(f"Updated ability level for {player}", level="INFO", extra_data={"level": selected_level})
        except Exception as e:
            logger.error(f"Error updating ability levels for {player}: {e}")
            _config_manager.log_to_json(f"Error updating ability levels for {player}: {str(e)}", level="ERROR")

    def update_temp_settings(self, key: str, value):
        try:
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
                    ai_config = _config_manager.load_ai_config()
                    if value in ai_config["ai_types"]:
                        self.temp_settings.ai_configs["player_1"]["ai_code"] = ai_config["ai_types"][value]["code"]
                        ai_specific_config = _config_manager.load_ai_specific_config(self.temp_settings.ai_configs["player_1"]["ai_code"])
                        self.temp_settings.ai_configs["player_1"]["params"] = ai_specific_config.get("player_1", DEFAULT_AI_PARAMS.copy())
            elif key == "player_2_ai_type":
                self.temp_settings.player_2_ai_type = value
                self.temp_settings.ai_configs["player_2"] = {
                    "ai_type": value,
                    "ai_code": None,
                    "ability_level": self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5),
                    "params": {}
                }
                if value != "none":
                    ai_config = _config_manager.load_ai_config()
                    if value in ai_config["ai_types"]:
                        self.temp_settings.ai_configs["player_2"]["ai_code"] = ai_config["ai_types"][value]["code"]
                        ai_specific_config = _config_manager.load_ai_specific_config(self.temp_settings.ai_configs["player_2"]["ai_code"])
                        self.temp_settings.ai_configs["player_2"]["params"] = ai_specific_config.get("player_2", DEFAULT_AI_PARAMS.copy())
            elif key == "ai_pause_time":
                self.temp_settings.ai_pause_time = value
            _config_manager.log_to_json(f"Updated temp setting: {key}", level="INFO", extra_data={"value": value})
        except Exception as e:
            logger.error(f"Error updating temp settings for {key}: {e}")
            _config_manager.log_to_json(f"Error updating temp settings for {key}: {str(e)}", level="ERROR")

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

            ai_config = _config_manager.load_ai_config()
            ai_config["ai_configs"]["player_1"].update(self.temp_settings.ai_configs["player_1"])
            ai_config["ai_configs"]["player_2"].update(self.temp_settings.ai_configs["player_2"])
            _config_manager.save_ai_config(ai_config)

            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_saved"]
            )
            _config_manager.log_to_json("Advanced settings saved", level="INFO")
        except Exception as e:
            self.log_error(f"خطا در ذخیره تنظیمات پیشرفته: {str(e)}")
            _config_manager.log_to_json(f"Error saving advanced settings: {str(e)}", level="ERROR")

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

    def choose_color(self, color_var: tk.StringVar, button_key: str):
        try:
            current_color = color_var.get()
            color = colorchooser.askcolor(color=current_color, title=LANGUAGES[self.settings.language]["choose_color"],
                                         parent=self.window)
            if color[1]:
                color_var.set(color[1])
                if button_key == "p1_color_button":
                    self.update_temp_settings("player_1_color", hex_to_rgb(color[1]))
                elif button_key == "p2_color_button":
                    self.update_temp_settings("player_2_color", hex_to_rgb(color[1]))
                elif button_key == "b1_color_button":
                    self.update_temp_settings("board_color_1", hex_to_rgb(color[1]))
                elif button_key == "b2_color_button":
                    self.update_temp_settings("board_color_2", hex_to_rgb(color[1]))
                self.update_color_button(getattr(self, button_key), color[1])
                _config_manager.log_to_json(f"Color chosen for {button_key}", level="INFO", extra_data={"color": color[1]})
        except Exception as e:
            self.log_error(f"خطا در انتخاب رنگ: {str(e)}")
            _config_manager.log_to_json(f"Error choosing color for {button_key}: {str(e)}", level="ERROR")

    @staticmethod
    def update_color_button(button: ttk.Button, color: str):
        try:
            button.configure(style="Custom.TButton")
            button.configure(text=f"{color}")
        except Exception as e:
            logger.error(f"Error updating color button: {e}")
            _config_manager.log_to_json(f"Error updating color button: {str(e)}", level="ERROR")

    def upload_image(self, key: str, var: tk.StringVar = None):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")],
                parent=self.window
            )
            if file_path:
                if var:
                    var.set(file_path)
                    self.entries[key] = var
                else:
                    self.entries[f"{key}_image"] = tk.StringVar(value=file_path)
                _config_manager.log_to_json(f"Image uploaded for {key}", level="INFO", extra_data={"file_path": file_path})
        except Exception as e:
            self.log_error(f"خطا در آپلود تصویر برای {key}: {str(e)}")
            _config_manager.log_to_json(f"Error uploading image for {key}: {str(e)}", level="ERROR")

    def update_piece_preview(self):
        try:
            self.preview_canvas.delete("all")
            style = self.piece_style_var.get()
            p1_color = self.player_1_color_var.get()
            if style == "circle":
                self.preview_canvas.create_oval(10, 10, 40, 40, fill=p1_color, outline="black")
            elif style == "outlined_circle":
                self.preview_canvas.create_oval(10, 10, 40, 40, outline=p1_color, width=2)
            elif style == "square":
                self.preview_canvas.create_rectangle(10, 10, 40, 40, fill=p1_color, outline="black")
            elif style == "diamond":
                self.preview_canvas.create_polygon(25, 10, 40, 25, 25, 40, 10, 25, fill=p1_color, outline="black")
            elif style == "star":
                points = [25, 10, 30, 20, 40, 20, 32, 28, 35, 40, 25, 32, 15, 40, 18, 28, 10, 20, 20, 20]
                self.preview_canvas.create_polygon(points, fill=p1_color, outline="black")
            elif style == "custom" and self.entries.get("player_1_piece"):
                img_path = self.entries["player_1_piece"].get()
                if Path(img_path).exists():
                    img = tk.PhotoImage(file=img_path).subsample(2, 2)
                    self.preview_canvas.create_image(25, 25, image=img)
                    self.preview_canvas.image = img
            _config_manager.log_to_json("Piece preview updated", level="INFO", extra_data={"style": style})
        except Exception as e:
            logger.error(f"Error updating piece preview: {e}")
            _config_manager.log_to_json(f"Error updating piece preview: {str(e)}", level="ERROR")

    @staticmethod
    def validate_string_input(input_str: str) -> str:
        try:
            return input_str.encode("utf-8").decode("utf-8")
        except UnicodeEncodeError:
            logger.error(f"Invalid string input: {input_str}")
            safe_str = input_str.encode("utf-8").decode("utf-8", errors="replace")
            _config_manager.log_to_json(f"Invalid string input processed: {safe_str}", level="WARNING")
            return safe_str

    def reset(self):
        try:
            self.copy_current_settings()
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

            for key in ["player_1_piece", "player_1_king", "player_2_piece", "player_2_king"]:
                self.entries[key].set(getattr(self.temp_settings, f"{key}_image"))

            for key in ["player_1_image", "player_2_image", "al1_image", "al2_image"]:
                self.entries[key].set(getattr(self.temp_settings, key))

            ability_mapping = {1: "very_weak", 3: "weak", 5: "medium", 7: "strong", 9: "very_strong"}
            p1_ability = self.temp_settings.ai_configs.get("player_1", {}).get("ability_level", 5)
            self.player_1_ability_var.set(LANGUAGES[self.settings.language][ability_mapping.get(p1_ability, "medium")])
            p2_ability = self.temp_settings.ai_configs.get("player_2", {}).get("ability_level", 5)
            self.player_2_ability_var.set(LANGUAGES[self.settings.language][ability_mapping.get(p2_ability, "medium")])

            self.update_piece_preview()
            self.update_color_button(self.p1_color_button, self.player_1_color_var.get())
            self.update_color_button(self.p2_color_button, self.player_2_color_var.get())
            self.update_color_button(self.b1_color_button, self.board_color_1_var.get())
            self.update_color_button(self.b2_color_button, self.board_color_2_var.get())
            self.toggle_ai_vs_ai_options()
            self.toggle_repeat_options()
            self.toggle_timer()

            messagebox.showinfo(
                LANGUAGES[self.settings.language]["info"],
                LANGUAGES[self.settings.language]["settings_reset"],
                parent=self.window
            )
            _config_manager.log_to_json("Settings reset to defaults", level="INFO")
        except Exception as e:
            self.log_error(f"خطا در بازنشانی تنظیمات: {str(e)}")
            _config_manager.log_to_json(f"Error resetting settings: {str(e)}", level="ERROR")

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

            ai_config = _config_manager.load_ai_config()
            for ai_type, var in [
                (self.player_1_ai_type_var.get(), LANGUAGES[self.settings.language]["player_1_ai"]),
                (self.player_2_ai_type_var.get(), LANGUAGES[self.settings.language]["player_2_ai"])
            ]:
                if ai_type != "none" and ai_type not in ai_config["ai_types"]:
                    self.log_error(f"{var}: نوع '{ai_type}' پیدا نشد")
                    return

            config = _config_manager.load_config()
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
                "player_1_name": self.validate_string_input(self.player_1_name_var.get()),
                "player_2_name": self.validate_string_input(self.player_2_name_var.get()),
                "al1_name": self.validate_string_input(self.al1_name_var.get()),
                "al2_name": self.validate_string_input(self.al2_name_var.get()),
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
            _config_manager.save_config(config)

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
                    ai_specific_config = _config_manager.load_ai_specific_config(code)
                    ai_specific_config[player] = self.temp_settings.ai_configs[player]["params"]
                    _config_manager.save_ai_specific_config(code, ai_specific_config)

            _config_manager.save_ai_config(ai_config)
            _config_manager.log_to_json("Settings saved successfully", level="INFO", extra_data={"config": config})

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
            _config_manager.log_to_json(f"Error saving settings: {str(e)}", level="ERROR")

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


#interface.py:
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.text = text
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 16)
        self.enabled = True
        log_to_json(
            f"Button initialized: {text}",
            level="INFO",
            extra_data={"x": x, "y": y, "width": width, "height": height}
        )

    def draw(self, screen):
        try:
            color = self.color if self.enabled else (self.color[0] // 2, self.color[1] // 2, self.color[2] // 2)
            pygame.draw.rect(screen, color, self.rect, border_radius=5)
            pygame.draw.rect(screen, (0, 0, 0), self.rect, 2, border_radius=5)
            text_surface = self.font.render(self.text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=self.rect.center)
            screen.blit(text_surface, text_rect)
            # لاگ فقط در صورت تغییر حالت دکمه
            if not hasattr(self, '_last_enabled') or self._last_enabled != self.enabled:
                log_to_json(
                    f"Drew button: {self.text}",
                    level="INFO",
                    extra_data={"enabled": self.enabled, "rect": [self.rect.x, self.rect.y, self.rect.width, self.rect.height]}
                )
                self._last_enabled = self.enabled
        except Exception as e:
            log_to_json(
                f"Error drawing button {self.text}: {str(e)}",
                level="ERROR",
                extra_data={"rect": [self.rect.x, self.rect.y, self.rect.width, self.rect.height]}
            )

    def is_clicked(self, pos):
        clicked = self.rect.collidepoint(pos)
        log_to_json(
            f"Button {self.text} clicked: {'clicked' if clicked else 'not clicked'}",
            level="INFO",  # فقط هنگام کلیک
            extra_data={"pos": list(pos)}
        )
        return clicked

class GameInterface:
    def __init__(self, settings):
        try:
            self.settings = settings
            self.config = load_config()
            self.WINDOW_WIDTH = self.config['window_width']
            self.WINDOW_HEIGHT = self.config['window_height']
            self.BOARD_WIDTH = self.config['board_width']
            self.PANEL_WIDTH = self.config['panel_width']
            self.MENU_HEIGHT = self.config['menu_height']
            self.BORDER_THICKNESS = self.config['border_thickness']
            self.SQUARE_SIZE = self.config['square_size']
            self.BUTTON_SPACING_FROM_BOTTOM = self.config['button_spacing_from_bottom']
            self.PLAYER_IMAGE_SIZE = self.config['player_image_size']
            self.ANIMATION_FRAMES = self.config.get('animation_frames', 20)
            self.BLACK = (0, 0, 0)
            self.LIGHT_GRAY = (200, 200, 200)
            self.SKY_BLUE = (135, 206, 235)
            self.LIGHT_GREEN = (144, 238, 144)
            self.BLUE = (0, 0, 255)
            self.WHITE = (255, 255, 255)
            self.GRAY = (128, 128, 128)

            # تنظیم رنگ‌ها از config
            for key, value in self.config.items():
                if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                    value = hex_to_rgb(value)
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

            self.root = tk.Tk()
            self.root.withdraw()
            self.game = Game(settings, self)
            self.screen = self.game.screen
            self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
            self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)
            self.last_update = pygame.time.get_ticks()

            # تعریف دکمه‌ها
            self.new_game_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 100,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[settings.language]["start_game"], self.SKY_BLUE
            )
            self.reset_scores_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 50,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[self.settings.language]["reset_scores"], self.LIGHT_GREEN
            )
            self.pause_button = Button(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20,
                self.WINDOW_HEIGHT - self.BUTTON_SPACING_FROM_BOTTOM - 150,
                self.PANEL_WIDTH - 40, 40,
                LANGUAGES[settings.language]["pause"], self.LIGHT_GRAY
            )

            self.player_1_move_ready = True
            self.player_2_move_ready = True
            self.player_1_pause_start = None
            self.player_2_pause_start = None
            self.pending_settings = None
            self.current_hand = 0
            self.pause_start_time = None
            self.auto_start_triggered = False
            self.player_1_image_surface = None
            self.player_2_image_surface = None
            self.player_1_piece_surface = None
            self.player_1_king_surface = None
            self.player_2_piece_surface = None
            self.player_2_king_surface = None
            self.load_player_images()
            self.load_piece_images()
            self._move_start_time = None
            self.hint_enabled_p1 = self.config['hint_enabled_p1_default']
            self.hint_enabled_p2 = self.config['hint_enabled_p2_default']
            self.hint_buttons = []
            self.hint_blink_timer = 0
            self.hint_blink_state = True
            self.undo_buttons = []

            log_to_json(
                "GameInterface initialized",
                level="INFO",
                extra_data={"settings": vars(self.settings), "window_size": [self.WINDOW_WIDTH, self.WINDOW_HEIGHT]}
            )
        except Exception as e:
            log_to_json(
                f"Error initializing GameInterface: {str(e)}",
                level="ERROR",
                extra_data={"settings": vars(self.settings) if self.settings else None}
            )
            raise

    def apply_pending_settings(self):
        try:
            if self.pending_settings:
                for key, value in self.pending_settings.items():
                    if key in ("player_1_color", "player_2_color", "board_color_1", "board_color_2"):
                        value = hex_to_rgb(value)
                    setattr(self.settings, key, value)
                self.pending_settings = None
                self.load_player_images()
                self.load_piece_images()
                self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                self.reset_scores_button.text = LANGUAGES[self.settings.language]["reset_scores"]
                self.pause_button.text = LANGUAGES[self.settings.language]["pause"]
                self.game = Game(self.settings, self)
                self.screen = self.game.screen
                self.current_hand = 0
                self.auto_start_triggered = False
                log_to_json(
                    "Applied pending settings",
                    level="INFO",
                    extra_data={"settings": vars(self.settings)}
                )
        except Exception as e:
            log_to_json(
                f"Error applying pending settings: {str(e)}",
                level="ERROR",
                extra_data={"pending_settings": self.pending_settings}
            )

    def load_player_images(self):
        player_1_image_path = None  # مقداردهی اولیه
        player_2_image_path = None  # مقداردهی اولیه
        try:
            player_1_image_path = None  # مقداردهی اولیه
            player_2_image_path = None  # مقداردهی اولیه
            player_1_image_path = (
                self.settings.player_1_image
                if self.settings.game_mode in ["human_vs_human", "human_vs_ai"]
                else self.settings.al1_image
            )
            player_2_image_path = (
                self.settings.player_2_image
                if self.settings.game_mode == "human_vs_human"
                else self.settings.al2_image
            )
            # ادامه کد همانند بالا...
        except Exception as e:
            log_to_json(
                f"Error in load_player_images: {str(e)}",
                level="ERROR",
                extra_data={"player_1_image": player_1_image_path, "player_2_image": player_2_image_path}
            )

    def load_piece_images(self):
        try:
            piece_size = self.SQUARE_SIZE - 20
            for path, attr in [
                (self.settings.player_1_piece_image, 'player_1_piece_surface'),
                (self.settings.player_1_king_image, 'player_1_king_surface'),
                (self.settings.player_2_piece_image, 'player_2_piece_surface'),
                (self.settings.player_2_king_image, 'player_2_king_surface')
            ]:
                surface = None
                path_obj = None  # مقداردهی اولیه
                if path:
                    path_obj = project_root / path
                    if path_obj.exists() and path_obj.stat().st_size > 0:
                        try:
                            image = pygame.image.load(str(path_obj))
                            if image.get_size() != (0, 0):
                                image = pygame.transform.scale(image, (piece_size, piece_size))
                                mask = pygame.Surface((piece_size, piece_size), pygame.SRCALPHA)
                                pygame.draw.circle(
                                    mask, (255, 255, 255, 255),
                                    (piece_size // 2, piece_size // 2),
                                    piece_size // 2
                                )
                                image.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
                                surface = image
                            else:
                                log_to_json(
                                    f"Invalid image file {path}: Empty or corrupted",
                                    level="ERROR",
                                    extra_data={"path": str(path_obj)}
                                )
                                messagebox.showerror(
                                    LANGUAGES[self.settings.language]["error"],
                                    f"Invalid image file {path}: Empty or corrupted",
                                    parent=self.root
                                )
                        except pygame.error as e:
                            log_to_json(
                                f"Error loading image {path}: {str(e)}",
                                level="ERROR",
                                extra_data={"path": str(path_obj)}
                            )
                            messagebox.showerror(
                                LANGUAGES[self.settings.language]["error"],
                                f"Error loading image {path}: {e}",
                                parent=self.root
                            )
                setattr(self, attr, surface)
                log_to_json(
                    f"Loaded piece image for {attr}: {'success' if surface else 'default'}",
                    level="INFO",
                    extra_data={"path": str(path_obj) if path_obj else None}
                )
        except Exception as e:
            log_to_json(
                f"Error in load_piece_images: {str(e)}",
                level="ERROR",
                extra_data={"paths": [self.settings.player_1_piece_image, self.settings.player_1_king_image,
                                      self.settings.player_2_piece_image, self.settings.player_2_king_image]}
            )

    def draw_piece(self, screen, piece_value, row, col, is_removal=False, is_kinged=False):
        try:
            if piece_value == 0:
                return
            draw_x = col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            draw_y = row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            radius = self.SQUARE_SIZE // 2 - 10
            is_player_2 = piece_value < 0
            is_king = abs(piece_value) == 2 or is_kinged
            base_color = self.settings.player_2_color if is_player_2 else self.settings.player_1_color

            if is_removal:
                removal_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(removal_surface, (50, 50, 50, 100), (radius, radius), radius)
                screen.blit(removal_surface, (draw_x - radius, draw_y - radius))
                log_to_json(
                    f"Drew removal piece at ({row}, {col})",
                    level="INFO",
                    extra_data={"position": [row, col], "is_kinged": is_kinged}
                )
                return

            shadow_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surface, (50, 50, 50, 100), (radius + 5, radius + 5), radius)
            screen.blit(shadow_surface, (draw_x - radius - 5, draw_y - radius - 5))

            piece_surface = (
                self.player_2_king_surface if is_king else self.player_2_piece_surface
            ) if is_player_2 else (
                self.player_1_king_surface if is_king else self.player_1_piece_surface
            )

            if piece_surface is not None:
                screen.blit(piece_surface, (draw_x - radius, draw_y - radius))
            else:
                gradient_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                for r in range(radius, 0, -1):
                    t = r / radius
                    color = tuple(int(c * t + 255 * (1 - t)) for c in base_color)
                    if self.settings.piece_style == "circle":
                        pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                    elif self.settings.piece_style == "outlined_circle":
                        pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                    elif self.settings.piece_style == "square":
                        pygame.draw.rect(gradient_surface, color, (radius - r, radius - r, r * 2, r * 2))
                    elif self.settings.piece_style == "diamond":
                        points = [(radius, radius - r), (radius + r, radius), (radius, radius + r), (radius - r, radius)]
                        pygame.draw.polygon(gradient_surface, color, points)
                    elif self.settings.piece_style == "star":
                        points = [
                            (radius, radius - r),
                            (radius + r * 0.3, radius - r * 0.3),
                            (radius + r, radius),
                            (radius + r * 0.3, radius + r * 0.3),
                            (radius, radius + r),
                            (radius - r * 0.3, radius + r * 0.3),
                            (radius - r, radius),
                            (radius - r * 0.3, radius - r * 0.3)
                        ]
                        pygame.draw.polygon(gradient_surface, color, points)
                if self.settings.piece_style in ["outlined_circle", "diamond", "star"]:
                    if self.settings.piece_style == "outlined_circle":
                        pygame.draw.circle(gradient_surface, self.BLACK, (radius, radius), radius, 2)
                    elif self.settings.piece_style == "diamond":
                        points = [(radius, radius - radius), (radius + radius, radius), (radius, radius + radius),
                                  (radius - radius, radius)]
                        pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                    elif self.settings.piece_style == "star":
                        points = [
                            (radius, radius - radius),
                            (radius + radius * 0.3, radius - radius * 0.3),
                            (radius + radius, radius),
                            (radius + radius * 0.3, radius + radius * 0.3),
                            (radius, radius + radius),
                            (radius - radius * 0.3, radius + radius * 0.3),
                            (radius - radius, radius),
                            (radius - radius * 0.3, radius - radius * 0.3)
                        ]
                        pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                shine_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                for r in range(radius // 2, 0, -1):
                    alpha = int(100 * (r / (radius / 2)))
                    pygame.draw.circle(shine_surface, (255, 255, 255, alpha), (radius - radius // 4, radius - radius // 4),
                                       r)
                gradient_surface.blit(shine_surface, (0, 0))
                screen.blit(gradient_surface, (draw_x - radius, draw_y - radius))
                if is_king:
                    crown_radius = radius // 2
                    pygame.draw.circle(screen, self.GRAY, (draw_x, draw_y), crown_radius)
                    pygame.draw.circle(screen, self.BLACK, (draw_x, draw_y), crown_radius, 1)

                log_to_json(
                    f"Drew piece at ({row}, {col})",
                    level="INFO",
                    extra_data={"piece_value": piece_value, "is_kinged": is_kinged, "is_player_2": is_player_2}
                )
        except Exception as e:
            log_to_json(
                f"Error drawing piece at ({row}, {col}): {str(e)}",
                level="ERROR",
                extra_data={"piece_value": piece_value, "row": row, "col": col}
            )

    def animate_move(self, piece_value, start_row, start_col, end_row, end_col, is_removal=False, is_kinged=False):
        try:
            log_to_json(
                f"Animating move: piece={piece_value}, from ({start_row}, {start_col}) to ({end_row}, {end_col})",
                level="INFO",
                extra_data={"is_removal": is_removal, "is_kinged": is_kinged}
            )
            if piece_value == 0 or piece_value is None:
                log_to_json(
                    "Invalid piece_value! Animation aborted",
                    level="WARNING",
                    extra_data={"piece_value": piece_value}
                )
                return

            start_x = start_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            start_y = start_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            end_x = end_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            end_y = end_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            radius = self.SQUARE_SIZE // 2 - 10

            is_player_2 = piece_value < 0
            is_king = abs(piece_value) == 2 or is_kinged
            base_color = self.settings.player_2_color if is_player_2 else self.settings.player_1_color

            piece_surface = (
                self.player_2_king_surface if is_king else self.player_2_piece_surface
            ) if is_player_2 else (
                self.player_1_king_surface if is_king else self.player_1_piece_surface
            )

            if is_removal:
                for frame in range(self.ANIMATION_FRAMES + 1):
                    t = frame / self.ANIMATION_FRAMES
                    alpha = int(255 * (1 - t))
                    self.draw_game()
                    removal_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(removal_surface, (*base_color, alpha), (radius, radius), radius)
                    self.screen.blit(removal_surface, (start_x - radius, start_y - radius))
                    self.game.draw_valid_moves()
                    pygame.display.update()
                    pygame.time.wait(20)
                    log_to_json(
                        f"Animating removal frame {frame}/{self.ANIMATION_FRAMES}",
                        level="DEBUG",
                        extra_data={"alpha": alpha}
                    )
            else:
                for frame in range(self.ANIMATION_FRAMES + 1):
                    t = frame / self.ANIMATION_FRAMES
                    current_x = start_x + (end_x - start_x) * (1 - np.cos(t * np.pi)) / 2
                    current_y = start_y + (end_y - start_y) * (1 - np.cos(t * np.pi)) / 2
                    self.draw_game()
                    shadow_surface = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
                    pygame.draw.circle(shadow_surface, (50, 50, 50, 100), (radius + 5, radius + 5), radius)
                    self.screen.blit(shadow_surface, (current_x - radius - 5, current_y - radius - 5))
                    if piece_surface is not None:
                        self.screen.blit(piece_surface, (current_x - radius, current_y - radius))
                    else:
                        gradient_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                        for r in range(radius, 0, -1):
                            t_r = r / radius
                            color = tuple(int(c * t_r + 255 * (1 - t_r)) for c in base_color)
                            if self.settings.piece_style == "circle":
                                pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                            elif self.settings.piece_style == "outlined_circle":
                                pygame.draw.circle(gradient_surface, color, (radius, radius), r)
                            elif self.settings.piece_style == "square":
                                pygame.draw.rect(gradient_surface, color, (radius - r, radius - r, r * 2, r * 2))
                            elif self.settings.piece_style == "diamond":
                                points = [(radius, radius - r), (radius + r, radius), (radius, radius + r),
                                          (radius - r, radius)]
                                pygame.draw.polygon(gradient_surface, color, points)
                            elif self.settings.piece_style == "star":
                                points = [
                                    (radius, radius - r),
                                    (radius + r * 0.3, radius - r * 0.3),
                                    (radius + r, radius),
                                    (radius + r * 0.3, radius + r * 0.3),
                                    (radius, radius + r),
                                    (radius - r * 0.3, radius + r * 0.3),
                                    (radius - r, radius),
                                    (radius - r * 0.3, radius - r * 0.3)
                                ]
                                pygame.draw.polygon(gradient_surface, color, points)
                        if self.settings.piece_style in ["outlined_circle", "diamond", "star"]:
                            if self.settings.piece_style == "outlined_circle":
                                pygame.draw.circle(gradient_surface, self.BLACK, (radius, radius), radius, 2)
                            elif self.settings.piece_style == "diamond":
                                points = [(radius, radius - radius), (radius + radius, radius), (radius, radius + radius),
                                          (radius - radius, radius)]
                                pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                            elif self.settings.piece_style == "star":
                                points = [
                                    (radius, radius - radius),
                                    (radius + radius * 0.3, radius - radius * 0.3),
                                    (radius + radius, radius),
                                    (radius + radius * 0.3, radius + radius * 0.3),
                                    (radius, radius + radius),
                                    (radius - radius * 0.3, radius + radius * 0.3),
                                    (radius - radius, radius),
                                    (radius - radius * 0.3, radius - radius * 0.3)
                                ]
                                pygame.draw.polygon(gradient_surface, self.BLACK, points, 2)
                        shine_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                        for r in range(radius // 2, 0, -1):
                            alpha = int(100 * (r / (radius / 2)))
                            pygame.draw.circle(shine_surface, (255, 255, 255, alpha),
                                               (radius - radius // 4, radius - radius // 4), r)
                        gradient_surface.blit(shine_surface, (0, 0))
                        self.screen.blit(gradient_surface, (current_x - radius, current_y - radius))
                        if is_kinged:
                            crown_radius = radius // 2
                            pygame.draw.circle(self.screen, self.GRAY, (current_x, current_y), crown_radius)
                            pygame.draw.circle(self.screen, self.BLACK, (current_x, current_y), crown_radius, 1)
                    self.game.draw_valid_moves()
                    pygame.display.update()
                    pygame.time.wait(20)
                    log_to_json(
                        f"Animating move frame {frame}/{self.ANIMATION_FRAMES}",
                        level="DEBUG",
                        extra_data={"current_position": [current_x, current_y]}
                    )
        except Exception as e:
            log_to_json(
                f"Error animating move: {str(e)}",
                level="ERROR",
                extra_data={
                    "piece_value": piece_value,
                    "start": [start_row, start_col],
                    "end": [end_row, end_col],
                    "is_removal": is_removal,
                    "is_kinged": is_kinged
                }
            )

    def draw_default_image(self, name, x, y):
        try:
            surface = pygame.Surface((self.PLAYER_IMAGE_SIZE, self.PLAYER_IMAGE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.LIGHT_GRAY,
                               (self.PLAYER_IMAGE_SIZE // 2, self.PLAYER_IMAGE_SIZE // 2),
                               self.PLAYER_IMAGE_SIZE // 2)
            text = self.small_font.render(name[:2], True, self.BLACK)
            text_rect = text.get_rect(center=(self.PLAYER_IMAGE_SIZE // 2, self.PLAYER_IMAGE_SIZE // 2))
            surface.blit(text, text_rect)
            self.screen.blit(surface, (x, y))
            log_to_json(
                f"Drew default image for {name} at ({x}, {y})",
                level="INFO",
                extra_data={"name": name, "position": [x, y]}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing default image for {name}: {str(e)}",
                level="ERROR",
                extra_data={"name": name, "position": [x, y]}
            )

    def draw_game(self):
        try:
            self.screen.fill(self.settings.board_color_1)
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, 0, self.WINDOW_WIDTH, self.MENU_HEIGHT))

            # ترسیم منو
            for text_key, x in [
                ("settings", 10),
                ("ai_progress", 120),
                ("help", 250),
                ("about_me", 350)
            ]:
                text_surface = self.small_font.render(LANGUAGES[self.settings.language][text_key], True, self.BLACK)
                self.screen.blit(text_surface, (x, 5))
                #log_to_json(
                    #f"Drew menu text: {text_key}",
                    #level="DEBUG",
                    #extra_data={"text": text_key, "position": [x, 5]}
                #)

            pygame.draw.rect(self.screen, self.BLUE,
                             (0, self.MENU_HEIGHT, self.BOARD_WIDTH + self.BORDER_THICKNESS * 2,
                              self.BOARD_WIDTH + self.BORDER_THICKNESS * 3), self.BORDER_THICKNESS)
            self.game.board.draw(self.screen, self.settings.board_color_1, self.settings.board_color_2)

            # ترسیم مهره‌ها
            for row in range(8):
                for col in range(8):
                    piece_value = self.game.board.board[row, col]
                    if piece_value != 0:
                        self.draw_piece(self.screen, piece_value, row, col)

            # بررسی موقعیت ماوس و ترسیم حرکات معتبر
            mouse_pos = pygame.mouse.get_pos()
            hovered_moves = self.game.check_mouse_hover(mouse_pos)
            self.game.draw_valid_moves(hovered_moves)

            self.draw_side_panel()

            # ترسیم پاورقی
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, (0, self.WINDOW_HEIGHT - 30, self.WINDOW_WIDTH, 30))
            footer_text = self.small_font.render(LANGUAGES[self.settings.language]["footer"], True, self.BLACK)
            footer_rect = footer_text.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT - 15))
            self.screen.blit(footer_text, footer_rect)
            log_to_json(
                "Drew game interface",
                level="INFO",
                extra_data={"board_state": self.game.board.board.tolist()}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing game interface: {str(e)}",
                level="ERROR",
                extra_data={"board_state": self.game.board.board.tolist() if hasattr(self.game, 'board') else None}
            )

    def draw_side_panel(self):
        player_1_name = None  # مقداردهی اولیه
        player_2_name = None  # مقداردهی اولیه
        try:
            scale = 0.85
            panel_rect = pygame.Rect(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2,
                self.MENU_HEIGHT,
                self.PANEL_WIDTH,
                self.WINDOW_HEIGHT - self.MENU_HEIGHT - 30
            )
            pygame.draw.rect(self.screen, self.LIGHT_GRAY, panel_rect)
            y = self.MENU_HEIGHT + 20

            # عنوان نتیجه بازی
            result_title = self.font.render(LANGUAGES[self.settings.language]["game_result"], True, self.BLACK)
            result_rect = result_title.get_rect(
                center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y)
            )
            self.screen.blit(result_title, result_rect)
            y += int(40 * scale)

            # نمایش نتیجه بازی
            if self.game.game_over:
                if self.game.winner is None:
                    result_text = "بازی مساوی شد" if self.settings.language == "fa" else "Game is a Draw"
                elif not self.game.winner:
                    result_text = LANGUAGES[self.settings.language]["player_wins"]
                else:
                    result_text = LANGUAGES[self.settings.language]["player_wins"]
                result_display = self.small_font.render(result_text, True, self.BLACK)
                result_display_rect = result_display.get_rect(
                    center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y)
                )
                self.screen.blit(result_display, result_display_rect)
                log_to_json(
                    f"Drew game result: {result_text}",
                    level="INFO",
                    extra_data={"winner": self.game.winner}
                )
            y += int(70 * scale)

            # نام بازیکنان
            player_1_name = (
                self.settings.player_1_name
                if self.settings.game_mode in ["human_vs_human", "human_vs_ai"]
                else self.settings.al1_name
            )
            player_2_name = (
                self.settings.player_2_name
                if self.settings.game_mode == "human_vs_human"
                else self.settings.al2_name
            )

            # تصاویر بازیکنان
            scaled_image_size = int(self.PLAYER_IMAGE_SIZE * scale)
            if self.player_1_image_surface:
                scaled_image = pygame.transform.scale(self.player_1_image_surface, (scaled_image_size, scaled_image_size))
                self.screen.blit(
                    scaled_image,
                    (self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y)
                )
            else:
                self.draw_default_image(player_1_name, self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y)

            # متن "vs"
            vs_font = pygame.font.SysFont('Arial', int(24 * scale), bold=True)
            vs_text = vs_font.render("vs", True, self.BLUE)
            vs_text.set_alpha(200)
            vs_rect = vs_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2,
                    y + scaled_image_size // 2
                )
            )
            pygame.draw.rect(
                self.screen, self.LIGHT_GRAY,
                (vs_rect.x - int(5 * scale), vs_rect.y - int(5 * scale), vs_rect.width + int(10 * scale), vs_rect.height + int(10 * scale)),
                border_radius=int(5 * scale)
            )
            self.screen.blit(vs_text, vs_rect)

            # تصویر بازیکن دوم
            if self.player_2_image_surface:
                scaled_image = pygame.transform.scale(self.player_2_image_surface, (scaled_image_size, scaled_image_size))
                self.screen.blit(
                    scaled_image,
                    (
                        self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size,
                        y
                    )
                )
            else:
                self.draw_default_image(
                    player_2_name,
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size,
                    y
                )

            y += int(10 * scale)

            # نام‌های بازیکنان
            player_1_name_text = self.small_font.render(player_1_name, True, self.BLACK)
            player_1_name_rect = player_1_name_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20 + scaled_image_size // 2,
                    y + scaled_image_size + int(10 * scale)
                )
            )
            self.screen.blit(player_1_name_text, player_1_name_rect)

            player_2_name_text = self.small_font.render(player_2_name, True, self.BLACK)
            player_2_name_rect = player_2_name_text.get_rect(
                center=(
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH - 20 - scaled_image_size // 2,
                    y + scaled_image_size + int(10 * scale)
                )
            )
            self.screen.blit(player_2_name_text, player_2_name_rect)

            y += int((scaled_image_size + 30) * scale)

            # جدول امتیازات
            table_rect = pygame.Rect(
                self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y, int((self.PANEL_WIDTH - 40) * scale), int(80 * scale)
            )
            shadow_rect = pygame.Rect(table_rect.x + int(5 * scale), table_rect.y + int(5 * scale), table_rect.width, table_rect.height)
            pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=int(10 * scale))

            gradient_surface = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
            for i in range(table_rect.height):
                color = (255 - i * 2, 255 - i * 2, 255 - i * 2)
                pygame.draw.line(gradient_surface, color, (0, i), (table_rect.width, i))
            mask = pygame.Surface((table_rect.width, table_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, table_rect.width, table_rect.height), border_radius=int(10 * scale))
            gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
            self.screen.blit(gradient_surface, (table_rect.x, table_rect.y))

            pygame.draw.rect(self.screen, self.BLACK, table_rect, 2, border_radius=int(10 * scale))
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x + table_rect.width // 3, table_rect.y),
                (table_rect.x + table_rect.width // 3, table_rect.y + table_rect.height), 2
            )
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x + 2 * table_rect.width // 3, table_rect.y),
                (table_rect.x + 2 * table_rect.width // 3, table_rect.y + table_rect.height), 2
            )
            pygame.draw.line(
                self.screen, self.BLACK,
                (table_rect.x, table_rect.y + table_rect.height // 2),
                (table_rect.x + table_rect.width, table_rect.y + table_rect.height // 2), 2
            )

            wins_1 = self.small_font.render(str(self.game.player_1_wins), True, self.BLACK)
            wins_1_rect = wins_1.get_rect(
                center=(table_rect.x + table_rect.width // 6, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_1, wins_1_rect)

            wins_text = self.small_font.render(LANGUAGES[self.settings.language]["wins"], True, self.BLACK)
            wins_text_rect = wins_text.get_rect(
                center=(table_rect.x + table_rect.width // 2, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_text, wins_text_rect)

            wins_2 = self.small_font.render(str(self.game.player_2_wins), True, self.BLACK)
            wins_2_rect = wins_2.get_rect(
                center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + table_rect.height // 4)
            )
            self.screen.blit(wins_2, wins_2_rect)

            pieces_1 = self.small_font.render(str(self.game.board.player_1_left), True, self.BLACK)
            pieces_1_rect = pieces_1.get_rect(
                center=(table_rect.x + table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_1, pieces_1_rect)

            pieces_text = self.small_font.render(LANGUAGES[self.settings.language]["pieces"], True, self.BLACK)
            pieces_text_rect = pieces_text.get_rect(
                center=(table_rect.x + table_rect.width // 2, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_text, pieces_text_rect)

            pieces_2 = self.small_font.render(str(self.game.board.player_2_left), True, self.BLACK)
            pieces_2_rect = pieces_2.get_rect(
                center=(table_rect.x + 5 * table_rect.width // 6, table_rect.y + 3 * table_rect.height // 4)
            )
            self.screen.blit(pieces_2, pieces_2_rect)

            y += int(110 * scale)

            # عنوان زمان
            time_title = self.small_font.render(LANGUAGES[self.settings.language]["time"], True, self.BLACK)
            time_title_rect = time_title.get_rect(
                center=(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + self.PANEL_WIDTH // 2, y))
            self.screen.blit(time_title, time_title_rect)
            y += int(20 * scale)

            # تایمر
            timer_rect = pygame.Rect(self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 + 20, y, int((self.PANEL_WIDTH - 40) * scale), int(40 * scale))
            shadow_rect = pygame.Rect(timer_rect.x + int(5 * scale), timer_rect.y + int(5 * scale), timer_rect.width, timer_rect.height)
            pygame.draw.rect(self.screen, (100, 100, 100), shadow_rect, border_radius=int(10 * scale))
            gradient_surface = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
            for i in range(timer_rect.height):
                color = (135 - i * 2, 206 - i * 2, 235 - i * 2)
                pygame.draw.line(gradient_surface, color, (0, i), (timer_rect.width, i))
            mask = pygame.Surface((timer_rect.width, timer_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, timer_rect.width, timer_rect.height), border_radius=int(10 * scale))
            gradient_surface.blit(mask, (0, 0), None, pygame.BLEND_RGBA_MULT)
            self.screen.blit(gradient_surface, (timer_rect.x, timer_rect.y))

            pygame.draw.rect(self.screen, self.BLACK, timer_rect, 2, border_radius=int(10 * scale))
            pygame.draw.line(self.screen, self.BLACK, (timer_rect.x + timer_rect.width // 2, timer_rect.y),
                             (timer_rect.x + timer_rect.width // 2, timer_rect.y + timer_rect.height), 2)

            bold_font = pygame.font.SysFont('Arial', int(18 * scale), bold=True)
            player_1_time = self.game.timer.get_current_time(False)
            player_2_time = self.game.timer.get_current_time(True)
            timer_1 = bold_font.render(f"{int(player_1_time)} s", True, self.BLACK if not self.game.turn else self.GRAY)
            timer_1_rect = timer_1.get_rect(
                center=(timer_rect.x + timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
            self.screen.blit(timer_1, timer_1_rect)

            timer_2 = bold_font.render(f"{int(player_2_time)} s", True, self.BLACK if self.game.turn else self.GRAY)
            timer_2_rect = timer_2.get_rect(
                center=(timer_rect.x + 3 * timer_rect.width // 4, timer_rect.y + timer_rect.height // 2))
            self.screen.blit(timer_2, timer_2_rect)

            y += int(60 * scale)

            # دکمه‌های Undo و Redo
            undo_button_x = (
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 +
                    (self.PANEL_WIDTH - int(2 * self.config['undo_button_width'] * scale) - int(self.config['undo_redo_button_spacing'] * scale)) // 2
            )
            undo_button_y = y + int(self.config['undo_redo_y_offset'] * scale)

            undo_button = Button(
                undo_button_x, undo_button_y, int(self.config['undo_button_width'] * scale), int(self.config['undo_button_height'] * scale),
                LANGUAGES[self.settings.language]["undo"], self.SKY_BLUE
            )
            undo_button.enabled = len(self.game.history.move_history) >= 2

            redo_button = Button(
                undo_button_x + int(self.config['undo_button_width'] * scale) + int(self.config['undo_redo_button_spacing'] * scale), undo_button_y,
                int(self.config['redo_button_width'] * scale), int(self.config['redo_button_height'] * scale),
                LANGUAGES[self.settings.language]["redo"], self.SKY_BLUE
            )
            redo_button.enabled = len(self.game.history.redo_stack) > 0

            self.undo_buttons = [undo_button, redo_button]
            for button in self.undo_buttons:
                button.draw(self.screen)

            y += int((self.config['undo_button_height'] + self.config['undo_redo_y_offset'] + 10) * scale)

            # دکمه‌های Hint
            hint_button_x = (
                    self.BOARD_WIDTH + self.BORDER_THICKNESS * 2 +
                    (self.PANEL_WIDTH - int(2 * self.config['hint_button_width'] * scale) - int(self.config['hint_button_spacing'] * scale)) // 2
            )
            hint_button_y = y + int(self.config['hint_button_y_offset'] * scale)

            p1_hint_text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p1 else "hint_off"]
            p1_hint_button = Button(
                hint_button_x, hint_button_y, int(self.config['hint_button_width'] * scale), int(self.config['hint_button_height'] * scale),
                p1_hint_text, self.SKY_BLUE
            )

            p2_hint_text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p2 else "hint_off"]
            p2_hint_button = Button(
                hint_button_x + int(self.config['hint_button_width'] * scale) + int(self.config['hint_button_spacing'] * scale), hint_button_y,
                int(self.config['hint_button_width'] * scale), int(self.config['hint_button_height'] * scale),
                p2_hint_text, self.SKY_BLUE
            )

            self.hint_buttons = [p1_hint_button, p2_hint_button]

            if self.settings.game_mode == "human_vs_ai" and not self.game.turn:
                p1_hint_button.enabled = True
                p2_hint_button.enabled = False
            elif self.settings.game_mode == "human_vs_ai" and self.game.turn:
                p1_hint_button.enabled = False
                p2_hint_button.enabled = True
            elif self.settings.game_mode == "human_vs_human":
                p1_hint_button.enabled = True
                p2_hint_button.enabled = True
            else:
                p1_hint_button.enabled = False
                p2_hint_button.enabled = False

            for button in self.hint_buttons:
                button.draw(self.screen)

            y += int((self.config['hint_button_height'] + self.config['hint_button_y_offset'] + 10) * scale)

            # دکمه‌های بازی جدید، بازنشانی امتیازات و توقف
            self.pause_button.draw(self.screen)
            self.new_game_button.draw(self.screen)
            self.reset_scores_button.draw(self.screen)

            log_to_json(
                "Drew side panel",
                level="INFO",
                extra_data={"player_1_name": player_1_name, "player_2_name": player_2_name}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing side panel: {str(e)}",
                level="ERROR",
                extra_data={"player_1_name": player_1_name, "player_2_name": player_2_name}
            )

    def _handle_events(self, settings_window, ai_progress_window, help_window, about_window):
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    log_to_json("Quit event received", level="INFO")
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    log_to_json(
                        f"Mouse click at {pos}",
                        level="DEBUG",
                        extra_data={"pos": list(pos)}
                    )
                    if pos[1] < self.MENU_HEIGHT:
                        if not (settings_window.is_open or ai_progress_window.is_open or
                                help_window.is_open or about_window.is_open):
                            if pos[0] < 120:
                                logger.info("Opening settings_window")
                                settings_window.create_widgets()
                                settings_window.is_open = True
                            elif 120 <= pos[0] < 250:
                                logger.info("Opening ai_progress_window")
                                ai_progress_window.create_widgets()
                                ai_progress_window.is_open = True
                            elif 250 <= pos[0] < 350:
                                logger.info("Opening help_window")
                                help_window.create_widgets()
                                help_window.is_open = True
                            elif 350 <= pos[0] < 450:
                                logger.info("Opening about_window")
                                about_window.create_widgets()
                                about_window.is_open = True
                    elif self.new_game_button.is_clicked(pos):
                        logger.info("New game button clicked")
                        if not self.game.game_started:
                            logger.info("Starting new game")
                            self.game.start_game()
                            self.new_game_button.text = LANGUAGES[self.settings.language]["new_game"]
                            self.last_update = pygame.time.get_ticks()
                            self.current_hand = 0
                            self.auto_start_triggered = False
                            sound_path = project_root / "assets" / "start.wav"
                            if self.settings.sound_enabled and sound_path.exists() and sound_path.stat().st_size > 0:
                                try:
                                    pygame.mixer.Sound(str(sound_path)).play()
                                    log_to_json("Played start sound", level="INFO", extra_data={"sound_path": str(sound_path)})
                                except pygame.error as e:
                                    log_to_json(
                                        f"Error playing start.wav: {str(e)}",
                                        level="ERROR",
                                        extra_data={"sound_path": str(sound_path)}
                                    )
                        elif self.game.game_started and not self.game.game_over:
                            confirmed = messagebox.askyesno(
                                LANGUAGES[self.settings.language]["warning"],
                                LANGUAGES[self.settings.language]["new_game_warning"],
                                parent=self.root
                            )
                            if confirmed:
                                logger.info("Restarting game")
                                self.game.game_over = True
                                self.game.score_updated = False
                                self.game.init_game()
                                self.game.game_started = False
                                self.game.game_over = False
                                self.game.winner = None
                                self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                                self.last_update = pygame.time.get_ticks()
                                self.current_hand = 0
                                self.auto_start_triggered = False
                                pygame.event.clear()
                        elif self.game.game_over:
                            logger.info("Starting new game after game over")
                            self.game.game_over = True
                            self.game.score_updated = False
                            self.game.init_game()
                            self.game.game_started = False
                            self.game.game_over = False
                            self.game.winner = None
                            self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                            self.last_update = pygame.time.get_ticks()
                            self.current_hand = 0
                            self.auto_start_triggered = False
                            pygame.event.clear()
                    elif self.reset_scores_button.is_clicked(pos):
                        logger.info("Reset scores button clicked")
                        if messagebox.askyesno(
                                LANGUAGES[self.settings.language]["warning"],
                                LANGUAGES[self.settings.language]["reset_scores_warning"],
                                parent=self.root
                        ):
                            self.game.player_1_wins = 0
                            self.game.player_2_wins = 0
                            save_stats({"player_1_wins": 0, "player_2_wins": 0})
                            log_to_json(
                                "Scores reset",
                                level="INFO",
                                extra_data={"player_1_wins": 0, "player_2_wins": 0}
                            )
                    elif self.pause_button.is_clicked(pos) and self.game.game_started and not self.game.game_over:
                        logger.info("Pause button clicked")
                        self.game.paused = not self.game.paused
                        if self.game.paused:
                            self.game.timer.pause()
                            self.pause_button.text = LANGUAGES[self.settings.language]["resume"]
                            log_to_json("Game paused", level="INFO")
                        else:
                            self.game.timer.unpause()
                            self.pause_button.text = LANGUAGES[self.settings.language]["pause"]
                            log_to_json("Game resumed", level="INFO")
                    elif self.game.game_started:
                        for button in self.undo_buttons:
                            if hasattr(button, 'enabled') and button.enabled and button.is_clicked(pos):
                                if button.text == LANGUAGES[self.settings.language]["undo"]:
                                    self.undo_move()
                                elif button.text == LANGUAGES[self.settings.language]["redo"]:
                                    self.redo_move()
                        for button in self.hint_buttons:
                            if hasattr(button, 'enabled') and button.enabled and button.is_clicked(pos):
                                if button == self.hint_buttons[0]:
                                    self.toggle_hint_p1()
                                elif button == self.hint_buttons[1]:
                                    self.toggle_hint_p2()
                        logger.debug("Handling board click")
                        self.game.handle_click(pos)
            return True
        except Exception as e:
            log_to_json(
                f"Error handling events: {str(e)}",
                level="ERROR",
                extra_data={"event_type": None}
            )
            return True

    def update(self):
        try:
            current_time = pygame.time.get_ticks()
            self.game.update_game()
            if self.game.game_started and not self.game.game_over:
                if self.settings.game_mode == "ai_vs_ai":
                    if self.player_1_move_ready and not self.game.turn:
                        logger.debug("Player 1 AI move ready")
                        if "ai_1" in self.game.ai_players:
                            self.player_1_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_1")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_1",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_1",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_1_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_1")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_1",
                                level="ERROR",
                                extra_data={"ai_id": "ai_1", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif self.player_2_move_ready and self.game.turn:
                        logger.debug("Player 2 AI move ready")
                        if "ai_2" in self.game.ai_players:
                            self.player_2_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_2")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_2",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_2",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_2_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_2")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_2",
                                level="ERROR",
                                extra_data={"ai_id": "ai_2", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif not self.player_1_move_ready and not self.game.turn:
                        if current_time - self._move_start_time >= self.settings.ai_pause_time:
                            logger.debug("Player 1 AI move ready after pause")
                            self.player_1_move_ready = True
                    elif not self.player_2_move_ready and self.game.turn:
                        if current_time - self._move_start_time >= self.settings.ai_pause_time:
                            logger.debug("Player 2 AI move ready after pause")
                            self.player_2_move_ready = True
                elif self.settings.game_mode == "human_vs_ai" and self.game.turn:
                    if self.player_2_move_ready:
                        logger.debug("Player 2 AI move ready (human_vs_ai)")
                        if "ai_2" in self.game.ai_players:
                            self.player_2_move_ready = False
                            self._move_start_time = current_time
                            success = self.game.make_ai_move("ai_2")
                            if not success:
                                log_to_json(
                                    "AI move failed for ai_2 (human_vs_ai)",
                                    level="ERROR",
                                    extra_data={
                                        "ai_id": "ai_2",
                                        "turn": self.game.turn,
                                        "board": self.game.board.board.tolist(),
                                        "game_mode": self.settings.game_mode
                                    }
                                )
                                self.game.change_turn()
                                self.player_2_move_ready = True
                            else:
                                logger.info("AI move succeeded for ai_2 (human_vs_ai)")
                                self.game.change_turn()
                        else:
                            log_to_json(
                                "No AI available for ai_2 (human_vs_ai)",
                                level="ERROR",
                                extra_data={"ai_id": "ai_2", "turn": self.game.turn}
                            )
                            self.game.change_turn()
                    elif current_time - self._move_start_time >= self.settings.ai_pause_time:
                        logger.debug("Player 2 AI move ready after pause (human_vs_ai)")
                        self.player_2_move_ready = True
                if self.game.game_over and self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game":
                    if self.current_hand < self.settings.repeat_hands:
                        if self.pause_start_time is None:
                            self.pause_start_time = current_time
                        elif current_time - self.pause_start_time >= self.settings.pause_between_hands:
                            logger.info("Starting new hand")
                            self.game.reset_board()
                            self.game.start_game()
                            self.current_hand += 1
                            self.pause_start_time = None
                            self.auto_start_triggered = False
                    else:
                        logger.info("All hands completed")
                        self.auto_start_triggered = True
                        self.new_game_button.text = LANGUAGES[self.settings.language]["start_game"]
                        self.game.game_started = False
            self.draw_game()
            pygame.display.update()
            log_to_json(
                "Game updated",
                level="DEBUG",
                extra_data={"game_started": self.game.game_started, "game_over": self.game.game_over}
            )
        except Exception as e:
            log_to_json(
                f"Error updating game: {str(e)}",
                level="ERROR",
                extra_data={"game_started": self.game.game_started, "game_over": self.game.game_over}
            )

    def close_windows(self, settings_window, ai_progress_window, help_window, about_window):
        try:
            for window in [settings_window, ai_progress_window, help_window, about_window]:
                if window is not None and window.is_open:
                    window.close()
                    log_to_json(
                        f"Closed window: {window.__class__.__name__}",
                        level="INFO"
                    )
            self.root.destroy()
            log_to_json("Tkinter root destroyed", level="INFO")
        except tk.TclError as e:
            log_to_json(
                f"TclError closing windows: {str(e)}",
                level="ERROR"
            )

    def undo_move(self):
        try:
            if len(self.game.history.move_history) < 1:
                logger.info("No moves to undo")
                return
            self.game.history.undo(self.game)
            if self.game.multi_jump_active and self.game.selected:
                valid_moves = self.game.get_valid_moves(*self.game.selected)
                self.game.valid_moves = {(move[2], move[3]): skipped for move, skipped in valid_moves.items() if skipped}
            elif not self.game.multi_jump_active:
                self.game.valid_moves = {}
            self.game.draw_valid_moves()
            log_to_json(
                "Undo move performed",
                level="INFO",
                extra_data={"move_history_length": len(self.game.history.move_history)}
            )
        except Exception as e:
            log_to_json(
                f"Error undoing move: {str(e)}",
                level="ERROR",
                extra_data={"move_history_length": len(self.game.history.move_history)}
            )

    def redo_move(self):
        try:
            self.game.history.redo(self.game)
            self.game.draw_valid_moves()
            log_to_json(
                "Redo move performed",
                level="INFO",
                extra_data={"redo_stack_length": len(self.game.history.redo_stack)}
            )
        except Exception as e:
            log_to_json(
                f"Error redoing move: {str(e)}",
                level="ERROR",
                extra_data={"redo_stack_length": len(self.game.history.redo_stack)}
            )

    def toggle_hint_p1(self):
        try:
            self.hint_enabled_p1 = not self.hint_enabled_p1
            self.hint_buttons[0].text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p1 else "hint_off"]
            log_to_json(
                f"Toggled hint for player 1: {self.hint_enabled_p1}",
                level="INFO",
                extra_data={"hint_enabled_p1": self.hint_enabled_p1}
            )
        except Exception as e:
            log_to_json(
                f"Error toggling hint for player 1: {str(e)}",
                level="ERROR"
            )

    def toggle_hint_p2(self):
        try:
            self.hint_enabled_p2 = not self.hint_enabled_p2
            self.hint_buttons[1].text = LANGUAGES[self.settings.language]["hint_on" if self.hint_enabled_p2 else "hint_off"]
            log_to_json(
                f"Toggled hint for player 2: {self.hint_enabled_p2}",
                level="INFO",
                extra_data={"hint_enabled_p2": self.hint_enabled_p2}
            )
        except Exception as e:
            log_to_json(
                f"Error toggling hint for player 2: {str(e)}",
                level="ERROR"
            )

    def draw_hint(self, start_row, start_col, end_row, end_col):
        try:
            if (start_row is None or start_col is None or end_row is None or end_col is None) or not (
                    self.hint_enabled_p1 and not self.game.turn or self.hint_enabled_p2 and self.game.turn):
                log_to_json(
                    "Skipping hint drawing: Invalid coordinates or hint disabled",
                    level="DEBUG",
                    extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
                )
                return
            current_time = pygame.time.get_ticks()
            if current_time - self.hint_blink_timer > self.config['hint_blink_interval']:
                self.hint_blink_state = not self.hint_blink_state
                self.hint_blink_timer = current_time
            if not self.hint_blink_state:
                return
            start_x = start_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            start_y = start_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            end_x = end_col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.BORDER_THICKNESS
            end_y = end_row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2 + self.MENU_HEIGHT + self.BORDER_THICKNESS
            pygame.draw.circle(self.screen, self.config['hint_circle_color'], (start_x, start_y),
                               self.config['hint_circle_radius'])
            pygame.draw.circle(self.screen, self.config['hint_circle_color'], (end_x, end_y),
                               self.config['hint_circle_radius'])
            log_to_json(
                f"Drew hint from ({start_row}, {start_col}) to ({end_row}, {end_col})",
                level="DEBUG",
                extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
            )
        except Exception as e:
            log_to_json(
                f"Error drawing hint: {str(e)}",
                level="ERROR",
                extra_data={"start": [start_row, start_col], "end": [end_row, end_col]}
            )

    def run(self):
        settings_window = SettingsWindow(self, self.root)
        ai_progress_window = AIProgressWindow(self, self.root)
        help_window = HelpWindow(self, self.root)
        about_window = AboutWindow(self, self.root)

        clock = pygame.time.Clock()

        running = True
        while running:
            running = self._handle_events(settings_window, ai_progress_window, help_window,
                                          about_window)

            self.update()

            self.screen.fill(self.settings.board_color_1)
            self.draw_game()

            if self.settings.game_mode in ["human_vs_human", "human_vs_ai"] and (
                    self.hint_enabled_p1 or self.hint_enabled_p2):
                hint = self.game.get_hint()  # استفاده از get_hint از game.py
                if hint is not None:
                    start_row, start_col, end_row, end_col = hint
                    self.draw_hint(start_row, start_col, end_row, end_col)
                else:
                    self.draw_hint(None, None, None, None)

            pygame.display.flip()

            try:
                self.root.update()
            except tk.TclError:
                running = False

            clock.tick(self.config.get('fps', 60))

        self.close_windows(settings_window, ai_progress_window, help_window, about_window)
        pygame.quit()
        sys.exit()


#checkers_game.py:
class CheckersGame:
    """Manages the checkers game logic.

    Attributes:
        board (Board): Current board state.
        current_player (int): 1 for player 1, -1 for player 2.
        move_count (int): Number of moves made.
    """

    def __init__(self, settings: GameSettings | None = None):
        self.config = _config_manager.load_config()
        self.settings = settings if settings else GameSettings()
        self.board = Board(self.settings)
        self.current_player = 1
        self.move_count = 0
        log_to_json(
            "CheckersGame initialized",
            level="DEBUG",
            extra_data={
                "board_size": self.board.board_size,
                "game_mode": self.settings.game_mode
            }
        )

    def reset(self):
        """Resets the game to the initial state."""
        self.board = Board(self.settings)
        self.current_player = 1
        self.move_count = 0
        log_to_json("Game reset", level="WARNING")

    def copy(self) -> 'CheckersGame':
        """Returns a deep copy of the game state."""
        new_game = CheckersGame(self.settings)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_count = self.move_count
        return new_game

    def set_state(self, state: np.ndarray):
        """Sets the board state from an array.

        Args:
            state (np.ndarray): State array with shape (board_size, board_size).
        """
        if state.shape != (self.board.board_size, self.board.board_size):
            log_to_json(
                f"Invalid state shape: {state.shape}",
                level="ERROR",
                extra_data={"expected_shape": (self.board.board_size, self.board.board_size)}
            )
            raise CheckersError("Invalid state shape")
        self.board.board = state.copy()
        self.board.player_1_left = np.sum(self.board.board > 0)
        self.board.player_2_left = np.sum(self.board.board < 0)
        log_to_json("State set successfully", level="DEBUG")

    def get_state(self) -> np.ndarray:
        """Returns the current state as an array.

        Returns:
            np.ndarray: State array.
        """
        return self.board.board.copy()

    @staticmethod
    def get_action_size() -> int:
        """Returns the size of the action space.

        Returns:
            int: Number of possible actions based on board size.
        """
        board_size = _config_manager.load_config().get("board_size", 8)
        return (board_size * board_size // 2) ** 2

    def get_legal_moves(self) -> list[Tuple[int, int, int, int]]:
        """Returns all legal moves for the current player.

        Returns:
            list: List of moves as (from_row, from_col, to_row, to_col).
        """
        legal_moves = []
        jumps_exist = False
        # Check jumps
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * self.current_player > 0:
                    moves = get_piece_moves(self.board, row, col, self.current_player)
                    jumps = [(to_pos, skipped) for to_pos, skipped in moves.items() if skipped]
                    if jumps:
                        jumps_exist = True
                        legal_moves.extend([(row, col, to_row, to_col) for (to_row, to_col), _ in jumps])
        if jumps_exist:
            log_to_json(
                f"Legal jumps for player {self.current_player}: {legal_moves}",
                level="DEBUG"
            )
            return legal_moves
        # Check simple moves
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * self.current_player > 0:
                    moves = get_piece_moves(self.board, row, col, self.current_player)
                    legal_moves.extend(
                        [(row, col, to_row, to_col) for (to_row, to_col), skipped in moves.items() if not skipped])
        log_to_json(
            f"Legal moves for player {self.current_player}: {legal_moves}",
            level="DEBUG"
        )
        return legal_moves

    def make_move(self, move: Tuple[int, int, int, int]) -> 'CheckersGame':
        """Executes a move and updates the game state.

        Args:
            move (tuple): Move as (from_row, from_col, to_row, to_col).

        Returns:
            CheckersGame: Updated game instance.
        """
        try:
            from_row, from_col, to_row, to_col = move
            result = make_move(self.board, move, self.current_player)
            new_board, is_promotion, has_more_jumps = result
            if new_board is None:
                raise CheckersError(f"Invalid move: {move}")

            self.board = new_board
            self.move_count += 1
            is_jump = abs(to_row - from_row) == 2

            # Check for promotion
            is_promotion = (
                (self.current_player == 1 and to_row == 0 and self.board.board[to_row, to_col] == 2) or
                (self.current_player == -1 and to_row == self.board.board_size - 1 and self.board.board[to_row, to_col] == -2)
            )

            self.current_player = -self.current_player
            if is_jump and not is_promotion and has_more_jumps:
                jumps = get_piece_moves(self.board, to_row, to_col, self.current_player)
                jumps = [(to_pos, skipped) for to_pos, skipped in jumps.items() if skipped]
                if jumps:
                    self.current_player = -self.current_player
                    self.move_count -= 1
                    log_to_json(
                        f"Additional jumps available at ({to_row}, {to_col}): {jumps}",
                        level="INFO",
                        extra_data={"move": move}
                    )

            log_to_json(
                f"Made move: {move}",
                level="INFO",
                extra_data={"is_jump": is_jump, "is_promotion": is_promotion, "current_player": self.current_player}
            )
            return self
        except CheckersError as e:
            log_to_json(
                f"CheckersError in make_move: {str(e)}",
                level="ERROR",
                extra_data={"move": move}
            )
            raise
        except Exception as e:
            log_to_json(
                f"Unexpected error in make_move: {str(e)}",
                level="ERROR",
                extra_data={"move": move, "board": self.board.board.tolist()}
            )
            raise

    def is_terminal(self) -> bool:
        """Checks if the game is in a terminal state.

        Returns:
            bool: True if the game is over.
        """
        max_total_moves = self.config.get("max_total_moves", 100)
        return not self.get_legal_moves() or self.move_count >= max_total_moves

    def get_reward(self) -> float:
        """Returns the reward for the current state.

        Returns:
            float: Reward value.
        """
        if not self.is_terminal():
            return 0
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            return -1 if self.current_player == 1 else 1
        return 0

    def get_outcome(self) -> float:
        """Returns the game outcome.

        Returns:
            float: Outcome value.
        """
        return self.get_reward()


#checkers_core.py:
def get_piece_moves(board: Board, row: int, col: int, player: int) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Returns possible moves for a piece at (row, col) for the given player.
    Format: {(to_row, to_col): [skipped_positions]}
    """
    try:
        moves = {}
        if not (0 <= row < board.board_size and 0 <= col < board.board_size):
            log_to_json(
                f"Invalid position ({row}, {col})",
                level="ERROR",
                extra_data={"row": row, "col": col, "board_size": board.board_size}
            )
            return moves
        piece = board.board[row, col]
        if piece == 0 or (player == 1 and piece < 0) or (player == -1 and piece > 0):
            log_to_json(
                f"No valid piece at ({row}, {col}) for player {player}",
                level="DEBUG",
                extra_data={"piece": piece, "player": player}
            )
            return moves
        is_king = abs(piece) == 2
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else (
            [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        )
        for dr, dc in directions:
            # حرکت ساده
            r, c = row + dr, col + dc
            if 0 <= r < board.board_size and 0 <= c < board.board_size:  # بررسی محدوده قبل از دسترسی
                if board.board[r, c] == 0:
                    moves[(r, c)] = []
            # حرکت پرشی
            r, c = row + 2 * dr, col + 2 * dc
            mid_r, mid_c = row + dr, col + dc
            if (0 <= r < board.board_size and 0 <= c < board.board_size and
                0 <= mid_r < board.board_size and 0 <= mid_c < board.board_size):  # بررسی محدوده قبل از دسترسی
                if (board.board[r, c] == 0 and board.board[mid_r, mid_c] != 0 and
                    ((player == 1 and board.board[mid_r, mid_c] < 0) or
                     (player == -1 and board.board[mid_r, mid_c] > 0))):
                    moves[(r, c)] = [(mid_r, mid_c)]
        log_to_json(
            f"Calculated moves for piece at ({row}, {col}): {moves}",
            level="DEBUG",
            extra_data={"player": player, "piece": piece, "moves": list(moves.keys())}
        )
        return moves
    except Exception as e:
        log_to_json(
            f"Error in get_piece_moves: {str(e)}",
            level="ERROR",
            extra_data={"row": row, "col": col, "player": player}
        )
        return {}

def make_move(board: Board, move: Tuple[int, int, int, int], player_number: int) -> Tuple[Optional[Board], bool, bool]:
    """
    Executes a move and returns (new_board, is_promotion, has_more_jumps).
    """
    try:
        from_row, from_col, to_row, to_col = move
        if not (0 <= from_row < board.board_size and 0 <= from_col < board.board_size and
                0 <= to_row < board.board_size and 0 <= to_col < board.board_size):
            log_to_json(
                f"Invalid move coordinates: {move}",
                level="ERROR",
                extra_data={"move": move, "board_size": board.board_size}
            )
            return None, False, False
        piece = board.board[from_row, from_col]
        player = 1 if player_number == 1 else -1
        if piece == 0 or (player == 1 and piece < 0) or (player == -1 and piece > 0):
            log_to_json(
                f"Invalid piece at ({from_row}, {from_col}) for player {player}",
                level="ERROR",
                extra_data={"piece": piece, "player": player}
            )
            return None, False, False
        valid_moves = get_piece_moves(board, from_row, from_col, player)
        if (to_row, to_col) not in valid_moves:
            log_to_json(
                f"Move {move} not valid",
                level="ERROR",
                extra_data={"move": move, "valid_moves": valid_moves}
            )
            return None, False, False
        new_board = board.copy()
        new_board.board[to_row, to_col] = piece
        new_board.board[from_row, from_col] = 0
        is_jump = abs(to_row - from_row) == 2
        skipped = valid_moves[(to_row, to_col)]
        for skip_r, skip_c in skipped:
            new_board.board[skip_r, skip_c] = 0
            if player == 1:
                new_board.player_2_left -= 1
            else:
                new_board.player_1_left -= 1
        is_promotion = False
        if player == 1 and to_row == 0 and abs(piece) == 1:
            new_board.board[to_row, to_col] = 2
            is_promotion = True
        elif player == -1 and to_row == board.board_size - 1 and abs(piece) == 1:
            new_board.board[to_row, to_col] = -2
            is_promotion = True
        has_more_jumps = False
        if is_jump and not is_promotion:
            additional_jumps = get_piece_moves(new_board, to_row, to_col, player)
            has_more_jumps = any(abs(to_r - to_row) == 2 for to_r, to_c in additional_jumps.keys())
        #log_to_json(
            #f"Move {move} executed: is_jump={is_jump}, is_promotion={is_promotion}, has_more_jumps={has_more_jumps}",
            #level="INFO",
            #extra_data={"move": move, "player": player}
        #)
        return new_board, is_promotion, has_more_jumps
    except Exception as e:
        log_to_json(
            f"Error in make_move: {str(e)}",
            level="ERROR",
            extra_data={"move": move, "player_number": player_number}
        )
        return None, False, False

def get_legal_moves(board: Board, player: int) -> List[Tuple[int, int, int, int]]:
    """
    Returns all legal moves for the player in 4-tuple format.
    """
    try:
        moves = []
        for row in range(board.board_size):
            for col in range(board.board_size):
                if board.board[row, col] * player > 0:
                    piece_moves = get_piece_moves(board, row, col, player)
                    for (to_row, to_col), _ in piece_moves.items():
                        moves.append((row, col, to_row, to_col))
        log_to_json(
            f"Legal moves for player {player}: {moves}",
            level="DEBUG",
            extra_data={"player": player, "move_count": len(moves)}
        )
        return moves
    except Exception as e:
        log_to_json(
            f"Error in get_legal_moves: {str(e)}",
            level="ERROR",
            extra_data={"player": player}
        )
        return []


#game.py:
class Game:
    def __init__(self, settings: Optional[GameSettings] = None, interface=None):
        settings_dict = None
        assets_dir = None
        self.board_size = config.get("board_size", 8)
        try:
            self.config = config
            self.hovered_piece = None
            if isinstance(settings, GameSettings):
                self.settings = settings
            else:
                settings_dict = settings or {}
                for key, value in self.config.items():
                    if key not in settings_dict:
                        settings_dict[key] = value
                assets_dir = settings_dict.get("assets_dir", "assets")
                settings_dict["player_1_piece_image"] = settings_dict.get("player_1_piece_image") or str(
                    project_root / assets_dir / "white_piece.png")
                settings_dict["player_2_piece_image"] = settings_dict.get("player_2_piece_image") or str(
                    project_root / assets_dir / "black_piece.png")
                settings_dict["player_1_king_image"] = settings_dict.get("player_1_king_image") or str(
                    project_root / assets_dir / "white_king.png")
                settings_dict["player_2_king_image"] = settings_dict.get("player_2_king_image") or str(
                    project_root / assets_dir / "black_king.png")
                valid_settings_keys = {
                    "window_width", "window_height", "board_size", "game_mode", "player_1_ai_type",
                    "player_2_ai_type", "player_starts", "use_timer", "game_time", "ai_pause_time",
                    "repeat_hands", "ai_vs_ai_mode", "player_1_piece_image", "player_2_piece_image",
                    "player_1_king_image", "player_2_king_image"
                }
                filtered_settings = {k: v for k, v in settings_dict.items() if k in valid_settings_keys}
                logger.debug(f"Settings passed to GameSettings: {filtered_settings}")
                try:
                    self.settings = GameSettings(**filtered_settings)
                except TypeError as e:
                    logger.error(f"Error creating GameSettings: {e}, filtered_settings={filtered_settings}")
                    raise
                if not hasattr(self.settings, "game_mode") or not self.settings.game_mode:
                    setattr(self.settings, "game_mode", "ai_vs_ai")
                ai_config = load_ai_config()
                available_ai_types = list(ai_config.get("ai_types", {}).keys())
                logger.debug(f"Available AI types: {available_ai_types}")
                if not hasattr(self.settings, "player_1_ai_type") or not self.settings.player_1_ai_type:
                    setattr(self.settings, "player_1_ai_type", available_ai_types[0] if available_ai_types else "none")
                if not hasattr(self.settings, "player_2_ai_type") or not self.settings.player_2_ai_type:
                    setattr(self.settings, "player_2_ai_type", available_ai_types[0] if available_ai_types else "none")
            self.interface = interface
            self.screen = pygame.display.set_mode((
                getattr(self.settings, "window_width", 940),
                getattr(self.settings, "window_height", 720)
            ))
            pygame.display.set_caption('Checkers: Player 1 vs Player 2')
            self._init_fonts()
            self._init_stats()
            self.board = Board(self.settings)
            self.checkers_game = CheckersGame(settings=self.settings)
            self.timer = TimerManager(
                getattr(self.settings, "use_timer", True),
                getattr(self.settings, "game_time", 5)
            )
            self.history = MoveHistory()
            self.game_over = False
            self.game_started = False
            self.last_update_time = 0
            self.multi_jump_active = False
            self.repeat_count = 0
            self.selected = None
            self.turn = False
            self.valid_moves = {}
            self.winner = None
            self.consecutive_uniform_moves = {1: 0, 2: 0}
            self.total_moves = {1: 0, 2: 0}
            self.no_capture_or_king_moves = 0
            self.last_state = None
            self.last_action = None
            self.move_log = []
            self.click_log = []
            self.score_updated = False
            self.time_up = False
            self.paused = False
            self.ai_players: Dict[str, 'BaseAI'] = {}
            self._init_ai_players()
            logger.info("Game initialized")
            _config_manager.log_to_json("Game initialized successfully", level="INFO", extra_data={"game_mode": self.settings.game_mode})
        except Exception as e:
            _config_manager.log_to_json(
                f"Error initializing game: {str(e)}",
                level="ERROR",
                extra_data={
                    "settings_dict": settings_dict,
                    "assets_dir": assets_dir
                }
            )
            raise

    def _init_fonts(self):
        font_name = 'Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial'
        self.font = pygame.font.SysFont(font_name, 24)
        self.small_font = pygame.font.SysFont(font_name, 18)
        logger.info("Fonts initialized")

    def _init_stats(self):
        stats = load_stats()
        self.player_1_wins = stats.get("player_1_wins", 0)
        self.player_2_wins = stats.get("player_2_wins", 0)
        logger.info(f"Stats initialized: player_1_wins={self.player_1_wins}, player_2_wins={self.player_2_wins}")

    def _init_ai_players(self):
        try:
            ai_config = load_ai_config()
            self.ai_players = {}
            game_mode = getattr(self.settings, "game_mode", "human_vs_human")
            player_1_ai_type = getattr(self.settings, "player_1_ai_type", "none")
            player_2_ai_type = getattr(self.settings, "player_2_ai_type", "none")
            logger.debug(f"Game mode: {game_mode}, Player 1 AI type: {player_1_ai_type}, Player 2 AI type: {player_2_ai_type}")
            if game_mode in ["human_vs_ai", "ai_vs_ai"] or player_1_ai_type != "none" or player_2_ai_type != "none":
                self.reward_calculator = RewardCalculator(self.checkers_game)
                self.update_ai_players(ai_config)
            else:
                self.reward_calculator = None
            logger.info(f"AI players initialized: {list(self.ai_players.keys())}")
        except Exception as e:
            logger.error(f"Error initializing AI players: {e}")
            _config_manager.log_to_json(f"Error initializing AI players: {str(e)}", level="ERROR")
            self.ai_players = {}

    @staticmethod
    def _load_ai_classes(config_dict):
        ai_type_to_class = {}
        logger.debug(f"Project root: {project_root}")
        for ai_type, ai_info in config_dict.get("ai_types", {}).items():
            class_name = ""  # تعریف پیش‌فرض
            try:
                module_name = ai_info.get("module", "")  # انتظار: modules.advanced_ai
                class_name = ai_info.get("class", "")
                logger.debug(f"Processing AI type: {ai_type}, module: {module_name}, class: {class_name}")
                module_path_parts = module_name.split(".")
                if len(module_path_parts) < 2 or module_path_parts[0] != "modules":
                    logger.error(f"Invalid module name format: {module_name}, expected 'modules.<module>'")
                    _config_manager.log_to_json(
                        f"Invalid module name format: {module_name}",
                        level="ERROR",
                        extra_data={"ai_type": ai_type}
                    )
                    continue
                module_file_name = module_path_parts[-1]
                module_path = project_root / "modules" / f"{module_file_name}.py"
                logger.debug(f"Checking module path: {module_path}")
                if not module_path.exists():
                    logger.error(f"Module file {module_path} not found")
                    _config_manager.log_to_json(
                        f"Module file {module_path} not found",
                        level="ERROR",
                        extra_data={"ai_type": ai_type}
                    )
                    continue
                if str(project_root) not in sys.path:
                    sys.path.append(str(project_root))
                    logger.debug(f"Added {project_root} to sys.path")
                logger.debug(f"Current sys.path: {sys.path}")
                module = importlib.import_module(module_name)
                if not hasattr(module, class_name):
                    logger.error(f"Class '{class_name}' not found in module {module_name}")
                    _config_manager.log_to_json(
                        f"Class '{class_name}' not found in module {module_name}",
                        level="ERROR",
                        extra_data={"ai_type": ai_type}
                    )
                    continue
                ai_class = getattr(module, class_name)
                if not isinstance(ai_class, type) or not issubclass(ai_class, BaseAI) or ai_class == BaseAI:
                    logger.error(f"{class_name} is not a valid AI class")
                    _config_manager.log_to_json(
                        f"{class_name} is not a valid AI class",
                        level="ERROR",
                        extra_data={"ai_type": ai_type}
                    )
                    continue
                ai_type_to_class[ai_type] = ai_class
                logger.info(f"Successfully mapped {ai_type} to {class_name}")
            except ImportError as e:
                logger.error(f"ImportError loading AI class {class_name} for type {ai_type}: {e}")
                _config_manager.log_to_json(
                    f"ImportError loading AI class {class_name} for type {ai_type}: {str(e)}",
                    level="ERROR",
                    extra_data={"ai_type": ai_type}
                )
            except Exception as e:
                logger.error(f"Unexpected error loading AI class {class_name} for type {ai_type}: {e}")
                _config_manager.log_to_json(
                    f"Unexpected error loading AI class {class_name} for type {ai_type}: {str(e)}",
                    level="ERROR",
                    extra_data={"ai_type": ai_type}
                )
        if not ai_type_to_class:
            logger.error("No valid AI classes loaded in _load_ai_classes")
            _config_manager.log_to_json("No valid AI classes loaded", level="ERROR")
        return ai_type_to_class

    def make_ai_move(self, ai_id: str) -> bool:
        move: Optional[Tuple[int, int, int, int]] = None
        valid_moves_formatted: List[Tuple[int, int, int, int]] = []
        try:
            ai = self.ai_players.get(ai_id)
            if not ai:
                _config_manager.log_to_json(
                    f"No AI found for {ai_id}. Available AI players: {list(self.ai_players.keys())}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "available_ai_players": list(self.ai_players.keys())}
                )
                return False
            player_number = 1 if ai_id == "ai_1" else 2
            player = 1 if player_number == 1 else -1
            logger.debug(f"Attempting AI move for {ai_id}, multi_jump_active={self.multi_jump_active}, selected={self.selected}")
            valid_moves = self._get_ai_valid_moves(ai)
            logger.debug(f"Valid moves for {ai_id}: {list(valid_moves.keys())}")
            if not valid_moves:
                logger.warning(f"No valid moves available for {ai_id}")
                _config_manager.log_to_json(
                    f"No valid moves available for {ai_id}",
                    level="WARNING",
                    extra_data={"ai_id": ai_id, "board": self.board.board.tolist()}
                )
                self.turn = not self.turn
                return False
            valid_moves_formatted = list(valid_moves.keys())
            move = ai.get_move(self.board.board)
            logger.debug(f"Move returned by {ai_id}: {move}")
            if not move or not isinstance(move, tuple) or len(move) != 4:
                logger.warning(f"Invalid move format returned by {ai_id}: {move}")
                _config_manager.log_to_json(
                    f"Invalid move format returned by {ai_id}: {move}",
                    level="WARNING",
                    extra_data={"ai_id": ai_id, "move": list(move) if move else None,
                                "valid_moves": [list(m) for m in valid_moves_formatted]}
                )
                move = valid_moves_formatted[0]
            elif move not in valid_moves_formatted:
                logger.warning(f"Move {move} not in valid moves for {ai_id}")
                _config_manager.log_to_json(
                    f"Move {move} not in valid moves for {ai_id}",
                    level="WARNING",
                    extra_data={"ai_id": ai_id, "move": list(move),
                                "valid_moves": [list(m) for m in valid_moves_formatted]}
                )
                move = valid_moves_formatted[0]
            from_row, from_col, to_row, to_col = move
            logger.debug(f"Validating move {move} for player {player}")
            board_before = self.board.copy()
            new_board, is_promotion, has_more_jumps = make_move(self.board, move, player_number)
            if new_board is None:
                logger.error(f"Invalid AI move: {move}")
                _config_manager.log_to_json(
                    f"Invalid AI move: {move}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "move": list(move),
                                "valid_moves": [list(m) for m in valid_moves_formatted]}
                )
                self.turn = not self.turn
                return False
            self.board = new_board
            self.checkers_game.set_state(self.board.board)
            board_after = self.board.copy()
            reward = self.reward_calculator.get_reward(player_number=player_number)
            logger.debug(f"Updating AI {ai_id} with move {move}, reward {reward}")
            ai.update(move, reward, board_before.board, board_after.board)
            is_jump = abs(to_row - from_row) == 2
            self._handle_multi_jump(to_row, to_col, is_jump, is_promotion, player_number)
            logger.info(f"AI move executed: {move}")
            if self.interface:
                piece_value = board_before.board[from_row, from_col]
                self.interface.animate_move(piece_value, from_row, from_col, to_row, to_col, is_kinged=is_promotion)
            return True
        except Exception as e:
            logger.error(f"Error in AI move for {ai_id}: {e}")
            _config_manager.log_to_json(
                f"Error in AI move for {ai_id}: {str(e)}",
                level="ERROR",
                extra_data={
                    "ai_id": ai_id,
                    "move": list(move) if move else None,
                    "valid_moves": [list(m) for m in valid_moves_formatted] if valid_moves_formatted else [],
                    "board": self.board.board.tolist() if hasattr(self, 'board') else None
                }
            )
            self.turn = not self.turn
            return False

    def _get_ai_valid_moves(self, ai) -> Dict[Tuple[int, int, int, int], List[Tuple[int, int]]]:
        try:
            player = 1 if not self.turn else -1
            if self.multi_jump_active and self.selected:
                moves = get_piece_moves(self.board, *self.selected, player)
                return {(self.selected[0], self.selected[1], to_row, to_col): skipped for (to_row, to_col), skipped in moves.items()}
            moves = ai.get_valid_moves(self.board.board)
            return {(move[0], move[1], move[2], move[3]): [] for move in moves if len(move) == 4}
        except Exception as e:
            logger.error(f"Error in _get_ai_valid_moves: {e}")
            _config_manager.log_to_json(
                f"Error in _get_ai_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": ai.ai_id}
            )
            return {}

    def get_hint(self):
        try:
            if self.game_over or (not self.turn and not self.config.get("hint_enabled_p1")) or (
                    self.turn and not self.config.get("hint_enabled_p2")):
                return None
            ai_id = "ai_1" if not self.turn else "ai_2"
            ai = self.ai_players.get(ai_id)
            if ai and self.settings.game_mode in ["human_vs_human", "human_vs_ai"]:
                if hasattr(ai, 'suggest_move'):
                    move = ai.suggest_move(self.board.board)
                else:
                    move = ai.get_move(self.board.board)
                if move and isinstance(move, tuple) and len(move) == 4:
                    return move
            return None
        except Exception as e:
            logger.error(f"Error getting hint: {e}")
            _config_manager.log_to_json(f"Error getting hint: {str(e)}", level="ERROR")
            return None

    def _handle_multi_jump(self, row, col, is_jump, is_promotion, player_number):
        try:
            self.valid_moves = {}
            self.multi_jump_active = False
            player = 1 if player_number == 1 else -1
            if not is_jump or is_promotion:
                self.selected = None
                self.turn = not self.turn
                logger.info(f"{'Simple move' if not is_jump else 'Jump move'}, turn switched to {'Player 2' if self.turn else 'Player 1'}")
            else:
                additional_jumps = get_piece_moves(self.board, row, col, player)
                jump_moves = {(row, col, to_row, to_col): skipped for (to_row, to_col), skipped in
                              additional_jumps.items() if abs(to_row - row) == 2}
                if jump_moves:
                    self.selected = (row, col)
                    self.valid_moves = jump_moves
                    self.multi_jump_active = True
                    logger.info(f"Multi-jump active for player {player_number} at ({row}, {col}) with moves: {list(jump_moves.keys())}")
                else:
                    self.selected = None
                    self.turn = not self.turn
                    logger.info(f"No additional jumps for player {player_number}, turn switched")
        except Exception as e:
            logger.error(f"Error in _handle_multi_jump: {e}")
            _config_manager.log_to_json(
                f"Error in _handle_multi_jump: {str(e)}",
                level="ERROR",
                extra_data={"player_number": player_number, "row": row, "col": col}
            )
            self.turn = not self.turn

    def change_turn(self):
        try:
            self.turn = not self.turn
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
            logger.info(f"Turn changed to {'Player 2' if self.turn else 'Player 1'}")
        except Exception as e:
            logger.error(f"Error changing turn: {e}")
            _config_manager.log_to_json(f"Error changing turn: {str(e)}", level="ERROR")

    def get_valid_moves(self, board: np.ndarray) -> Dict[Tuple[int, int, int, int], List[Tuple[int, int]]]:
        try:
            temp_game = self.checkers_game.copy()
            temp_game.set_state(board)
            temp_game.current_player = 1 if not self.turn else -1
            legal_moves = temp_game.get_legal_moves()
            valid_moves = {(move[0], move[1], move[2], move[3]): [] for move in legal_moves if len(move) == 4}
            logger.debug(f"Valid moves retrieved: {valid_moves}")
            return valid_moves
        except Exception as e:
            logger.error(f"Error in get_valid_moves: {e}")
            _config_manager.log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"board": board.tolist()}
            )
            raise CheckersError(f"Failed to get valid moves: {str(e)}")

    def update_ai_players(self, config_dict=None):
        try:
            config_dict = config_dict or load_ai_config()
            config_dict["game_settings"] = {
                "player_1_piece_image": getattr(self.settings, "player_1_piece_image", ""),
                "player_2_piece_image": getattr(self.settings, "player_2_piece_image", ""),
                "player_1_king_image": getattr(self.settings, "player_1_king_image", ""),
                "player_2_king_image": getattr(self.settings, "player_2_king_image", ""),
                "board_size": getattr(self.settings, "board_size", 8)
            }
            logger.debug(f"Game settings in config_dict: {config_dict['game_settings']}")
            ai_type_to_class = self._load_ai_classes(config_dict)
            if not ai_type_to_class:
                logger.error("No valid AI classes loaded in update_ai_players")
                _config_manager.log_to_json("No valid AI classes loaded", level="ERROR")
                return
            players = []
            game_mode = getattr(self.settings, "game_mode", "human_vs_human")
            player_1_ai_type = getattr(self.settings, "player_1_ai_type", "none")
            player_2_ai_type = getattr(self.settings, "player_2_ai_type", "none")
            logger.debug(
                f"Game mode: {game_mode}, Player 1 AI type: {player_1_ai_type}, Player 2 AI type: {player_2_ai_type}")
            if game_mode == "ai_vs_ai":
                players = ["player_1", "player_2"]
            elif game_mode == "human_vs_ai":
                players = ["player_2"]
            elif player_1_ai_type != "none":
                players = ["player_1"]
            elif player_2_ai_type != "none":
                players = ["player_2"]
            logger.debug(f"Players to initialize: {players}")
            self.ai_players = {}
            for player in players:
                ai_id = "ai_1" if player == "player_1" else "ai_2"
                ai_settings = config_dict["ai_configs"].get(player, {})
                ai_type = ai_settings.get("ai_type", player_1_ai_type if player == "player_1" else player_2_ai_type)
                model_name = ai_settings.get("ai_code", ai_type)
                # افزودن مقادیر پیش‌فرض برای پارامترهای مورد نیاز
                ai_settings.setdefault("ability_level", "default")
                ai_settings.setdefault("training_params", {})
                logger.debug(f"Initializing AI for {player}: ai_id={ai_id}, ai_type={ai_type}, model_name={model_name}")
                if ai_type and ai_type != "none":
                    ai_class = ai_type_to_class.get(ai_type)
                    if ai_class:
                        try:
                            self.ai_players[ai_id] = ai_class(
                                game=self.checkers_game,
                                model_name=model_name,
                                ai_id=ai_id,
                                training_params=ai_settings.get("training_params", {}),
                                ability_level=ai_settings.get("ability_level", "default")
                            )
                            logger.info(f"Loaded AI: {ai_type} for {player} (ID: {ai_id})")
                            _config_manager.log_to_json(
                                f"Loaded AI: {ai_type} for {player}",
                                level="INFO",
                                extra_data={"ai_id": ai_id, "ai_type": ai_type}
                            )
                        except Exception as e:
                            logger.error(f"Error loading AI {ai_type} for {ai_id}: {e}")
                            _config_manager.log_to_json(
                                f"Error loading AI {ai_type} for {ai_id}: {str(e)}",
                                level="ERROR",
                                extra_data={"ai_id": ai_id, "ai_type": ai_type}
                            )
                    else:
                        logger.error(f"No AI class found for type: {ai_type}")
                        _config_manager.log_to_json(
                            f"No AI class found for type: {ai_type}",
                            level="ERROR",
                            extra_data={"ai_type": ai_type}
                        )
        except Exception as e:
            logger.error(f"Error updating AI players: {e}")
            _config_manager.log_to_json(f"Error updating AI players: {str(e)}", level="ERROR")
            self.ai_players = {}

    def select(self, row, col):
        try:
            if self.paused or not self.game_started:
                logger.debug("Cannot select piece while paused or game not started")
                return False
            piece = self.board.board[row, col]
            player = 1 if not self.turn else -1
            logger.debug(f"Selecting piece at ({row}, {col}), piece={piece}, player={player}, game_mode={self.settings.game_mode}")
            if piece != 0 and piece * player > 0:
                valid_moves = get_piece_moves(self.board, row, col, player)
                if valid_moves:
                    self.selected = (row, col)
                    self.valid_moves = {(row, col, to_row, to_col): skipped for (to_row, to_col), skipped in
                                        valid_moves.items()}
                    logger.info(f"Selected piece at ({row}, {col}) with valid moves: {list(self.valid_moves.keys())}")
                    return True
                logger.debug(f"No valid moves for piece at ({row}, {col})")
            else:
                logger.debug(f"Invalid selection at ({row}, {col}): piece={piece}, player={player}")
            return False
        except Exception as e:
            logger.error(f"Error selecting piece at ({row}, {col}): {e}")
            _config_manager.log_to_json(
                f"Error selecting piece at ({row}, {col}): {str(e)}",
                level="ERROR",
                extra_data={"row": row, "col": col}
            )
            return False

    def _move(self, move: Tuple[int, int, int, int]):
        try:
            if self.paused or not self.game_started:
                logger.debug("Cannot move while paused or game not started")
                return False
            start_row, start_col, to_row, to_col = move
            if self.selected and (start_row, start_col) == self.selected and (to_row, to_col) in [(m[2], m[3]) for m in
                                                                                                  self.valid_moves]:
                player_number = 1 if not self.turn else 2
                self.history.save_state(self)
                new_board, is_promotion, has_more_jumps = make_move(self.board, move, player_number)
                if new_board is None:
                    logger.error(f"Invalid move: {move}")
                    return False
                self.board = new_board
                self.checkers_game.set_state(self.board.board)
                is_jump = abs(to_row - start_row) == 2
                self._update_move_counters(player_number, is_jump, is_promotion)
                self._handle_multi_jump(to_row, to_col, is_jump, is_promotion, player_number)
                logger.info(
                    f"Move {move} successful, player_1_left={self.board.player_1_left}, player_2_left={self.board.player_2_left}"
                )
                return True
            logger.error(f"Invalid move: {move}")
            return False
        except CheckersError as e:
            logger.error(f"CheckersError moving: {e}")
            return False
        except Exception as e:
            logger.error(f"Error moving: {e}")
            return False

    def _update_move_counters(self, player_number, is_capture, is_promotion):
        self.total_moves[player_number] += 1
        self.consecutive_uniform_moves[player_number] = 0 if is_capture or is_promotion else \
            self.consecutive_uniform_moves[player_number] + 1
        self.no_capture_or_king_moves = 0 if is_capture or is_promotion else self.no_capture_or_king_moves + 1

    def init_game(self):
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.timer.start_game()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")

    def start_game(self):
        try:
            self.game_started = True
            self.game_over = False
            self.winner = None
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
            self.last_update_time = pygame.time.get_ticks()
            self.timer.start_game()
            self.update_ai_players()
            self.repeat_count = self.settings.repeat_hands if self.settings.game_mode == "ai_vs_ai" and self.settings.ai_vs_ai_mode == "repeat_game" else 1
            logger.info("Game started")
        except Exception as e:
            logger.error(f"Error starting game: {e}")

    def reset_board(self):
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.checkers_game = CheckersGame(settings=self.settings)
            self.timer.start_game()
            self.update_ai_players()
            logger.info("Board reset")
        except Exception as e:
            logger.error(f"Error resetting board: {e}")

    def _reset_game_state(self):
        self.consecutive_uniform_moves = {1: 0, 2: 0}
        self.total_moves = {1: 0, 2: 0}
        self.multi_jump_active = False
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()
        self.selected = None
        self.turn = False if self.settings.player_starts else True
        self.valid_moves = {}
        self.game_over = False
        self.winner = None
        self.last_state = None
        self.last_action = None
        self.move_log = []
        self.click_log = []
        self.game_started = False
        self.score_updated = False
        self.repeat_count = 0
        self.time_up = False
        self.paused = False

    def check_mouse_hover(self, pos):
        try:
            if self.paused or not self.game_started or self.selected is not None:
                return None
            x, y = pos
            if x < self.config.get("board_width") and self._is_human_turn():
                row = (y - self.config.get("menu_height") - self.config.get("border_thickness")) // self.config.get(
                    "square_size")
                col = (x - self.config.get("border_thickness")) // self.config.get("square_size")
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    return None
                piece = self.board.board[row, col]
                player = 1 if not self.turn else -1
                if piece != 0 and piece * player > 0:
                    valid_moves = get_piece_moves(self.board, row, col, player)
                    if valid_moves:
                        self.hovered_piece = (row, col)
                        return {(row, col, to_row, to_col): skipped for (to_row, to_col), skipped in
                                valid_moves.items()}
                self.hovered_piece = None
            return None
        except Exception as e:
            logger.error(f"Error in check_mouse_hover: {e}")
            _config_manager.log_to_json(
                f"Error in check_mouse_hover: {str(e)}",
                level="ERROR",
                extra_data={"pos": pos}
            )
            return None

    def update_game(self):
        if not self.paused:
            try:
                self._update_game()
            except Exception as e:
                logger.error(f"Error updating game: {e}")
                _config_manager.log_to_json(
                    f"Error updating game: {str(e)}",
                    level="ERROR",
                    extra_data={"game_started": self.game_started, "game_over": self.game_over}
                )

    def _update_game(self):
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        try:
            if self.game_over:
                return self.winner
            player = 1 if not self.turn else -1
            for row in range(self.board.board_size):
                for col in range(self.board.board_size):
                    piece = self.board.board[row, col]
                    if piece != 0 and piece * player > 0:
                        moves = get_piece_moves(self.board, row, col, player)
                        if moves:
                            return None
            self.game_over = True
            self.winner = not self.turn
            logger.info(f"Game over, winner: {'black' if self.winner else 'white'}")
            return self.winner
        except Exception as e:
            logger.error(f"Error checking winner: {e}")
            _config_manager.log_to_json(
                f"Error checking winner: {str(e)}",
                level="ERROR"
            )
            return None

    def _has_valid_moves(self):
        player = 1 if not self.turn else -1
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * player > 0:
                    if get_piece_moves(self.board, row, col, player):
                        return True
        return False

    def _check_timer(self):
        player_1_time = self.timer.get_current_time(False)
        player_2_time = self.timer.get_current_time(True)
        if player_1_time <= 0:
            self._end_game(True)
            self.time_up = True
        elif player_2_time <= 0:
            self._end_game(False)
            self.time_up = True

    def _end_game(self, winner):
        self.winner = winner
        self.game_over = True
        if not self.score_updated:
            self.score_updated = True
            if self.winner is not None:
                if not self.winner:
                    self.player_1_wins += 1
                else:
                    self.player_2_wins += 1
            save_stats({"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins})
            logger.info(f"Game ended, winner: {self.winner}")
            _config_manager.log_to_json(
                f"Game ended, winner: {self.winner}",
                level="INFO",
                extra_data={"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins}
            )

    def handle_click(self, pos):
        if self.paused:
            logger.debug("Cannot handle click while paused")
            return False
        try:
            x, y = pos
            self.click_log.append((x, y))
            if len(self.click_log) > 50:
                self.click_log.pop(0)
            if x < self.config.get(
                    "board_width") and self.game_started and not self.game_over and self._is_human_turn():
                row = (y - self.config.get("menu_height") - self.config.get("border_thickness")) // self.config.get(
                    "square_size")
                col = (x - self.config.get("border_thickness")) // self.config.get("square_size")
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    logger.debug(f"Click outside board: ({row}, {col})")
                    return False
                logger.debug(f"Processing click at ({row}, {col}), turn={self.turn}, game_mode={self.settings.game_mode}")
                if self.selected:
                    move = (self.selected[0], self.selected[1], row, col)
                    if (move[2], move[3]) in [(m[2], m[3]) for m in self.valid_moves]:
                        self.history.save_state(self)
                        result = self._move(move)
                        logger.info(f"Move attempted: {move}, success={result}")
                        return result
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    player = 1 if not self.turn else -1
                    self.valid_moves = {(self.selected[0], self.selected[1], to_row, to_col): skipped for
                                        (to_row, to_col), skipped in
                                        get_piece_moves(self.board, *self.selected, player).items()}
                    logger.debug(f"Updated valid moves for multi-jump: {list(self.valid_moves.keys())}")
                logger.debug(f"Select result: {result}, selected={self.selected}")
                return result
            if self.game_over:
                self.selected = None
                self.valid_moves = {}
                self.multi_jump_active = False
                logger.debug("Game over, resetting selection")
            return False
        except Exception as e:
            logger.error(f"Error handling click at {pos}: {e}")
            _config_manager.log_to_json(
                f"Error handling click at {pos}: {str(e)}",
                level="ERROR",
                extra_data={"pos": pos, "game_mode": self.settings.game_mode}
            )
            return False

    def _is_human_turn(self):
        return self.settings.game_mode == "human_vs_human" or (
            self.settings.game_mode == "human_vs_ai" and not self.turn)

    def draw_valid_moves(self, hovered_moves=None):
        try:
            if not self.game_started or self.paused:
                return
            square_size = self.config.get("square_size", 80)
            border_thickness = self.config.get("border_thickness", 10)
            menu_height = self.config.get("menu_height", 100)
            valid_move_color = (0, 255, 0, 128)
            circle_radius = square_size // 4
            for move, _ in self.valid_moves.items():
                if not (isinstance(move, tuple) and len(move) == 4 and all(isinstance(x, int) for x in move)):
                    logger.error(f"Invalid move format in valid_moves: {move}")
                    continue
                _, _, row, col = move
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    logger.error(f"Invalid move position: ({row}, {col})")
                    continue
                x = border_thickness + col * square_size + square_size // 2
                y = menu_height + border_thickness + row * square_size + square_size // 2
                surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                pygame.draw.circle(surface, valid_move_color, (square_size // 2, square_size // 2), circle_radius)
                self.screen.blit(surface, (x - square_size // 2, y - square_size // 2))
                logger.debug(f"Drawing valid move at ({row}, {col})")
            if hovered_moves:
                for move, _ in hovered_moves.items():
                    if not (isinstance(move, tuple) and len(move) == 4 and all(isinstance(x, int) for x in move)):
                        logger.error(f"Invalid move format in hovered_moves: {move}")
                        continue
                    _, _, row, col = move
                    if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                        logger.error(f"Invalid hovered move position: ({row}, {col})")
                        continue
                    x = border_thickness + col * square_size + square_size // 2
                    y = menu_height + border_thickness + row * square_size + square_size // 2
                    surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    pygame.draw.circle(surface, valid_move_color, (square_size // 2, square_size // 2), circle_radius)
                    self.screen.blit(surface, (x - square_size // 2, y - square_size // 2))
                    logger.debug(f"Drawing hovered move at ({row}, {col})")
        except Exception as e:
            logger.error(f"Error in draw_valid_moves: {e}")
            _config_manager.log_to_json(
                f"Error in draw_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"valid_moves": [list(k) for k in self.valid_moves.keys()] if self.valid_moves else None,
                            "hovered_moves": [list(k) for k in hovered_moves.keys()] if hovered_moves else None}
            )

    def save_game_state(self):
        try:
            self.history.save_state(self)
            logger.info("Game state saved")
        except Exception as e:
            logger.error(f"Error saving game state: {e}")
            _config_manager.log_to_json(f"Error saving game state: {str(e)}", level="ERROR")


#base_ai.py:
class BaseAI(ABC):
    def __init__(self, game, model_name: str, ai_id: str, settings: Optional[Dict] = None):
        if not hasattr(game, 'copy') or not hasattr(game, 'get_legal_moves'):
            raise ValueError("Provided game object does not support required methods (copy, get_legal_moves)")
        self.game = game
        self.model_name = model_name
        self.ai_id = ai_id
        self.settings = settings or {}
        self.player_number = 1 if self.ai_id == "ai_1" else 2
        self.player = 1 if self.player_number == 1 else -1
        self.player_key = "player_1" if self.ai_id == "ai_1" else "player_2"

        # بارگذاری تنظیمات از ConfigManager (کش‌شده)
        try:
            config_data = _config_manager.load_ai_config()
            self.config = config_data["ai_configs"].get(self.player_key, self._default_config())
            if not isinstance(self.config, dict):
                logger.error(f"Invalid config format for {self.player_key}, using default")
                self.config = self._default_config()
        except Exception as e:
            logger.error(f"Error loading config for {self.ai_id}: {e}, using default")
            self.config = self._default_config()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # تنظیم مسیرهای مدل با استفاده از ConfigManager
        try:
            project_root = _config_manager.get_project_root()
            model_dir = self.config.get("model_dir", "models")
            pth_dir = project_root / model_dir
            pth_dir.mkdir(parents=True, exist_ok=True)
            self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
            self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
            self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"
        except Exception as e:
            logger.error(f"Error setting model paths for {self.ai_id}: {e}")
            raise CheckersError(f"Failed to set model paths: {str(e)}")

        # تنظیم سطح توانایی
        self.ability = min(max(int(self.config.get("ability_level", 1)), 1), 10)

        # متغیرهای آموزشی
        self.memory = deque(maxlen=self.config["training_params"]["memory_size"])
        self.gamma = self.config["training_params"]["gamma"]
        self.progress_tracker = None
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.config["training_params"]["update_target_every"]
        self.reward_calculator = RewardCalculator(game, ai_id=ai_id)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reward_threshold = self.config["training_params"]["reward_threshold"]

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        log_to_json(
            f"Initialized BaseAI for {ai_id} with model {model_name}",
            level="INFO",
            extra_data={"device": str(self.device), "ability": self.ability, "model_path": str(self.model_path)}
        )

    @staticmethod
    def _default_config():
        return {
            "ability_level": DEFAULT_AI_PARAMS.get("ability_level", 1),
            "training_params": {
                "memory_size": DEFAULT_AI_PARAMS.get("training_params", {}).get("memory_size", 10000),
                "gamma": DEFAULT_AI_PARAMS.get("training_params", {}).get("gamma", 0.99),
                "batch_size": DEFAULT_AI_PARAMS.get("training_params", {}).get("batch_size", 64),
                "update_target_every": DEFAULT_AI_PARAMS.get("training_params", {}).get("update_target_every", 1000),
                "reward_threshold": DEFAULT_AI_PARAMS.get("training_params", {}).get("reward_threshold", 1.0),
                "gradient_clip": DEFAULT_AI_PARAMS.get("training_params", {}).get("gradient_clip", 1.0),
                "target_update_alpha": DEFAULT_AI_PARAMS.get("training_params", {}).get("target_update_alpha", 0.005)
            },
            "network_params": {
                "board_size": DEFAULT_AI_PARAMS.get("network_params", {}).get("board_size", 8)
            },
            "advanced_nn_params": {
                "input_channels": DEFAULT_AI_PARAMS.get("advanced_nn_params", {}).get("input_channels", 3)
            }
        }

    def set_game(self, game):
        self.game = game
        if self.reward_calculator:
            self.reward_calculator.game = game
        log_to_json(f"Updated game object for {self.ai_id}", level="INFO")

    def get_move(self, board: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        try:
            valid_moves = self.get_valid_moves(board)
            if not valid_moves:
                log_to_json(
                    "No valid moves available",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id, "board": board.tolist()}
                )
                return None

            move = self.act(valid_moves)
            if not move or not isinstance(move, tuple) or len(move) != 4:
                log_to_json(
                    "act() returned invalid move, selecting first valid move",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id, "move": list(move) if move else None,
                                "valid_moves": [list(m) for m in valid_moves]}
                )
                move = valid_moves[0]

            log_to_json(
                f"Selected move: {move}",
                level="INFO",
                extra_data={"ai_id": self.ai_id}
            )
            return move
        except Exception as e:
            log_to_json(
                f"Error in get_move: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get move: {str(e)}")

    def get_valid_moves(self, board: np.ndarray) -> List[Tuple[int, int, int, int]]:
        try:
            temp_game = self.game.copy()
            temp_game.set_state(board)
            temp_game.current_player = self.player
            moves = temp_game.get_legal_moves()
            log_to_json(
                f"Retrieved valid moves: {moves}",
                level="DEBUG",
                extra_data={"ai_id": self.ai_id}
            )
            return moves
        except Exception as e:
            log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get valid moves: {str(e)}")

    def get_state(self, board: np.ndarray) -> torch.Tensor:
        try:
            board_size = self.game.board.board_size
            input_channels = self.config.get("advanced_nn_params", {}).get("input_channels", 3)
            state = np.zeros((input_channels, board_size, board_size), dtype=np.float32)
            for row in range(board_size):
                for col in range(board_size):
                    piece = board[row, col]
                    if piece != 0:
                        if piece > 0:
                            state[0, row, col] = 1
                            if abs(piece) == 2:
                                state[2, row, col] = 1
                        else:
                            state[1, row, col] = 1
                            if abs(piece) == 2:
                                state[2, row, col] = 1
            tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            log_to_json(
                f"State shape: {tensor.shape}",
                level="DEBUG",
                extra_data={"ai_id": self.ai_id}
            )
            return tensor
        except Exception as e:
            log_to_json(
                f"Error in get_state: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "board": board.tolist()}
            )
            raise CheckersError(f"Failed to get state: {str(e)}")

    @abstractmethod
    def act(self, valid_moves: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        pass

    def update(self, move: Tuple[int, int, int, int], reward: float, board_before: np.ndarray, board_after: np.ndarray):
        try:
            if move is None:
                log_to_json(
                    "Move is None, skipping update",
                    level="WARNING",
                    extra_data={"ai_id": self.ai_id}
                )
                return

            state = self.get_state(board_before)
            next_state = self.get_state(board_after)
            action = move
            done = self.game.game_over

            self.remember(state, action, reward, next_state, done)
            self.replay()
            self.update_target_network()
            self.save_model()
        except Exception as e:
            log_to_json(
                f"Error in update: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "move": move}
            )
            raise CheckersError(f"Failed to update AI: {str(e)}")

    def remember(self, state: torch.Tensor, action: Tuple[int, int, int, int], reward: float, next_state: torch.Tensor,
                 done: bool):
        try:
            self.current_episode_reward += reward
            self.memory.append((state, action, reward, next_state, done))
            self.save_important_experience(state, action, reward, next_state, done)
            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.current_episode_reward = 0
                with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                    json.dump(self.episode_rewards, f)
        except Exception as e:
            log_to_json(
                f"Error in remember: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "action": action}
            )
            raise CheckersError(f"Failed to store experience: {str(e)}")

    def replay(self):
        try:
            batch_size = self.config["training_params"]["batch_size"]
            if len(self.memory) < batch_size:
                return

            if self.policy_net is None or self.target_net is None or self.optimizer is None:
                raise CheckersError("Neural networks or optimizer not initialized")

            batch = list(self.memory)[-batch_size:]
            long_term_memory = self.load_long_term_memory()
            if long_term_memory:
                long_term_batch = sorted(long_term_memory, key=lambda x: x[2], reverse=True)[:batch_size // 2]
                batch = batch[:(batch_size // 2)] + long_term_batch

            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.cat(states)
            next_states = torch.cat(next_states)
            board_size = self.game.board.board_size
            action_indices = torch.tensor([
                (a[0] * board_size + a[1]) * (board_size * board_size) + (a[2] * board_size + a[3])
                for a in actions if a is not None
            ], device=self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

            current_q_values = self.policy_net(states).gather(1, action_indices.unsqueeze(1))
            with torch.no_grad():
                max_next_q_values = self.target_net(next_states).max(1)[0]
                target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(),
                self.config["training_params"]["gradient_clip"]
            )
            self.optimizer.step()

            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    (1 - self.config["training_params"]["target_update_alpha"]) * target_param.data +
                    self.config["training_params"]["target_update_alpha"] * policy_param.data
                )

            self.epoch += 1
            policy_loss = loss.item()
            player1_wins = sum(1 for r in self.episode_rewards if r > 0)
            player2_wins = sum(1 for r in self.episode_rewards if r < 0)
            self.progress_tracker.log_epoch(self.epoch, policy_loss, policy_loss, player1_wins, player2_wins)
        except Exception as e:
            log_to_json(
                f"Error in replay: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "batch_size": locals().get("batch_size", "undefined")}
            )
            raise CheckersError(f"Failed to replay experiences: {str(e)}")

    def update_target_network(self):
        try:
            if self.policy_net is None or self.target_net is None:
                raise CheckersError("Neural networks not initialized")

            self.episode_count += 1
            if self.episode_count % self.update_target_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                log_to_json(
                    f"Updated target network for {self.ai_id}",
                    level="INFO",
                    extra_data={"episode_count": self.episode_count}
                )
        except Exception as e:
            log_to_json(
                f"Error updating target network: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id}
            )
            raise CheckersError(f"Failed to update target network: {str(e)}")

    def save_model(self):
        try:
            if self.policy_net is None:
                raise CheckersError("Policy network not initialized")

            torch.save(self.policy_net.state_dict(), self.model_path)
            torch.save(self.policy_net.state_dict(), self.backup_path)
            with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                json.dump(self.episode_rewards, f)
            log_to_json(
                f"Saved model for {self.ai_id}",
                level="INFO",
                extra_data={"model_path": str(self.model_path)}
            )
        except Exception as e:
            log_to_json(
                f"Error saving model: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "model_path": str(self.model_path)}
            )
            raise CheckersError(f"Failed to save model: {str(e)}")

    def save_important_experience(self, state: torch.Tensor, action: Tuple[int, int, int, int], reward: float,
                                  next_state: torch.Tensor, done: bool):
        try:
            if reward > self.reward_threshold:
                with open(self.long_term_memory_path, "a") as f:
                    experience = {
                        "state": state.cpu().numpy().tolist(),
                        "action": action,
                        "reward": float(reward),
                        "next_state": next_state.cpu().numpy().tolist(),
                        "done": done,
                    }
                    json.dump(experience, f)
                    f.write("\n")
        except Exception as e:
            log_to_json(
                f"Error saving important experience: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "reward": reward}
            )
            raise CheckersError(f"Failed to save experience: {str(e)}")

    def load_long_term_memory(self) -> List[Tuple[torch.Tensor, Tuple[int, int, int, int], float, torch.Tensor, bool]]:
        try:
            if self.long_term_memory_path.exists():
                with open(self.long_term_memory_path, "r") as f:
                    experiences = []
                    for line in f.readlines():
                        exp = json.loads(line)
                        exp["state"] = torch.FloatTensor(exp["state"]).to(self.device)
                        exp["next_state"] = torch.FloatTensor(exp["next_state"]).to(self.device)
                        experiences.append((
                            exp["state"],
                            tuple(exp["action"]),
                            exp["reward"],
                            exp["next_state"],
                            exp["done"]
                        ))
                    return experiences
            return []
        except Exception as e:
            log_to_json(
                f"Error loading long-term memory: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": self.ai_id, "path": str(self.long_term_memory_path)}
            )
            raise CheckersError(f"Failed to load long-term memory: {str(e)}")

    @classmethod
    def get_metadata(cls) -> Dict[str, str]:
        return getattr(cls, "AI_METADATA", AI_METADATA)