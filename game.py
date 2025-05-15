import pygame
import numpy as np
from .board import Board
import importlib
from .timer import TimerManager
from .rewards import RewardCalculator
from .config import load_config, load_stats, save_stats, load_ai_config
from .constants import BOARD_WIDTH, MENU_HEIGHT, BORDER_THICKNESS, SQUARE_SIZE, RED


class Game:
    def __init__(self, settings, interface=None):
        #print("Game.__init__ called")
        config = load_config()
        self.board_size = config.get("network_params", {}).get("board_size", 8)
        self.max_no_capture_moves = config.get("max_no_capture_moves", 40)

        self.settings = settings
        self.interface = interface

        self.screen = pygame.display.set_mode((config.get("window_width", 940), config.get("window_height", 727)))
        pygame.display.set_caption('Checkers: Player 1 vs Player 2')

        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
        self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)

        stats = load_stats()
        self.player_1_wins = stats["player_1_wins"]
        self.player_2_wins = stats["player_2_wins"]

        self.ai_players = {}
        ai_id = None
        if self.settings.game_mode == "human_vs_ai":
            ai_id = "ai_2"  # فرض: بازیکن دوم هوش مصنوعی است
        elif self.settings.game_mode == "ai_vs_ai":
            ai_id = "ai_1"  # پیش‌فرض به بازیکن اول برای RewardCalculator
        elif self.settings.player_1_ai_type != "none":
            ai_id = "ai_1"
        elif self.settings.player_2_ai_type != "none":
            ai_id = "ai_2"

        if self.settings.game_mode in ["human_vs_ai", "ai_vs_ai"] or \
                self.settings.player_1_ai_type != "none" or \
                self.settings.player_2_ai_type != "none":
            self.reward_calculator = RewardCalculator(self)
        else:
            self.reward_calculator = None
        self.update_ai_players()

        self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)

        self.multi_jump_active = False
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()

        self.init_game()

    def update_ai_players(self):
        print(f"Updating AI players. Current ai_players: {self.ai_players}")
        config_dict = load_ai_config()

        ai_type_to_class = {}
        for ai_info in config_dict.get("available_ais", []):
            try:
                module_name = ai_info["module"]
                class_name = ai_info["class"]
                ai_type = ai_info["type"]
                print(f"Attempting to import {class_name} from {module_name} for type {ai_type}")
                module = importlib.import_module(module_name)
                ai_class = getattr(module, class_name)
                if not isinstance(ai_class, type):
                    print(f"Error: {class_name} is not a class, got {ai_class}")
                    continue
                ai_type_to_class[ai_type] = ai_class
                print(f"Successfully mapped {ai_type} to {class_name}")
            except Exception as e:
                print(f"Error importing AI class {class_name} from {module_name}: {e}")

        if not ai_type_to_class:
            print("Error: No valid AI classes loaded. Check available_ais in config.")
            return

        self.ai_players = {}
        if self.settings.game_mode == "ai_vs_ai":
            players = ["player_1", "player_2"]
        elif self.settings.game_mode == "human_vs_ai":
            players = ["player_2"]
        else:
            print(f"Error: Invalid game_mode: {self.settings.game_mode}")
            return

        for player in players:
            ai_id = "ai_1" if player == "player_1" else "ai_2"
            settings = config_dict["ai_configs"].get(player, {})
            ai_type = settings.get("ai_type")
            model_name = settings.get("ai_code", "default")
            print(f"Processing AI for {player} (ID: {ai_id}, type: {ai_type}, model_name: {model_name})")
            if ai_type and ai_type != "none":
                try:
                    ai_class = ai_type_to_class.get(ai_type)
                    if ai_class:
                        print(f"Attempting to load AI: {ai_type} for {player} (ID: {ai_id})")
                        self.ai_players[ai_id] = ai_class(game=self, model_name=model_name, ai_id=ai_id,
                                                          config=config_dict)
                        print(f"Loaded AI: {ai_type} for {player} (ID: {ai_id})")
                    else:
                        print(f"No AI class found for type: {ai_type}")
                except Exception as e:
                    print(f"Error loading AI {ai_type} for {player}: {e}")
            else:
                print(f"No AI type specified or ai_type is 'none' for {player}")

    def _create_ai(self, ai_type, model_name, ai_id):
        print(f"Creating AI: ai_type={ai_type}, model_name={model_name}, ai_id={ai_id}")
        if ai_type == "none":
            return None
        try:
            ai_config = load_ai_config()
            ai_info = next((ai for ai in ai_config["available_ais"] if ai["type"] == ai_type), None)
            if not ai_info:
                raise ValueError(f"AI type {ai_type} not found in ai_config.json")
            module = importlib.import_module(ai_info["module"])
            ai_class = getattr(module, ai_info["class"])
            config_dict = {"ai_configs": ai_config["ai_configs"]}
            return ai_class(game=self, model_name=model_name, ai_id=ai_id, config=config_dict)
        except Exception as e:
            print(f"Error creating AI {ai_id}: {e}")
            return None

    def init_game(self):
        print("init_game called")
        self.board = Board(self.settings)
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
        self.no_capture_or_king_moves = 0
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.repeat_count = 0

    def start_game(self):
        print("start_game called")
        self.game_started = True
        self.game_over = False
        self.winner = None
        self.selected = None
        self.valid_moves = {}
        self.multi_jump_active = False
        self.last_update_time = pygame.time.get_ticks()
        self.timer.start_game()
        self.update_ai_players()
        self.repeat_count = self.settings.repeat_hands if (
            self.settings.game_mode == "ai_vs_ai" and
            self.settings.ai_vs_ai_mode == "repeat_game"
        ) else 1

    def reset_board(self):
        print("reset_board called")
        self.board = Board(self.settings)
        self.selected = None
        self.turn = False if self.settings.player_starts else True
        self.valid_moves = {}
        self.game_over = False
        self.winner = None
        self.move_log = []
        self.click_log = []
        self.no_capture_or_king_moves = 0
        self.score_updated = False
        self.multi_jump_active = False
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.update_ai_players()

    def make_ai_move(self, ai_id=None):
        print(f"[make_ai_move] Called with ai_id: {ai_id}")
        print(f"[make_ai_move] Current ai_players: {self.ai_players}")
        print(
            f"[make_ai_move] Turn: {self.turn}, Multi jump active: {self.multi_jump_active}, Selected: {self.selected}")

        if ai_id is None:
            ai_id = "ai_1" if not self.turn else "ai_2"
            print(f"[make_ai_move] No ai_id provided, using turn: {self.turn} -> {ai_id}")

        if ai_id not in ["ai_1", "ai_2"] or ai_id not in self.ai_players:
            print(f"[make_ai_move] Invalid or missing AI for ai_id: {ai_id}")
            return False

        try:
            ai = self.ai_players[ai_id]
            print(f"[make_ai_move] AI object: {ai}, game: {ai.game}")
            print(f"[make_ai_move] Board shape: {self.board.board.shape}")
            board_before = self.board.board.copy()

            # محاسبه شماره بازیکن از ai_id
            player_number = 1 if ai_id == "ai_1" else 2
            print(f"[make_ai_move] Player number: {player_number}")

            # گرفتن حرکت‌های معتبر
            if self.multi_jump_active and self.selected:
                # فقط حرکات پرشی برای مهره انتخاب‌شده
                valid_moves = self.get_valid_moves(*self.selected)
                valid_moves = {(self.selected[0], self.selected[1], move[2], move[3]): skipped for move, skipped in
                               valid_moves.items() if skipped}
                print(f"[make_ai_move] Multi-jump moves for {self.selected}: {valid_moves}")
            else:
                valid_moves = ai.get_valid_moves(self.board.board)
            print(f"[make_ai_move] Valid moves from AI: {valid_moves}")

            if not valid_moves:
                print("[make_ai_move] No valid moves available")
                return False

            move = ai.get_move(self.board.board)
            print(f"[make_ai_move] AI move returned: {move}")
            if not move:
                print("[make_ai_move] Warning: AI returned None for move, selecting first valid move")
                move = list(valid_moves.keys())[0]

            # تجزیه حرکت
            from_row, from_col, to_row, to_col = move
            print(f"[make_ai_move] Attempting move from ({from_row}, {from_col}) to ({to_row}, {to_col})")

            # بررسی اینکه حرکت با self.selected در پرش چندگانه مطابقت دارد
            if self.multi_jump_active and self.selected and (from_row, from_col) != self.selected:
                print(f"[make_ai_move] Resetting self.selected to ({from_row}, {from_col})")
                self.selected = (from_row, from_col)

            # ذخیره حالت تخته قبل از حرکت
            self.interface.move_history.append((
                self.board.board.copy(),
                self.turn,
                valid_moves.copy(),
                self.board.player_1_left,
                self.board.player_2_left
            ))
            self.interface.redo_stack.clear()  # پاک کردن redo_stack

            # انتخاب مهره
            print(f"[make_ai_move] Selecting piece at ({from_row}, {from_col})")
            success = self.select(from_row, from_col)
            if success:
                print(f"[make_ai_move] Selection successful, moving to ({to_row}, {to_col})")
                # انجام حرکت
                success = self._move(to_row, to_col)
                if success:
                    board_after = self.board.board.copy()
                    reward = self.reward_calculator.get_reward(player_number=player_number)
                    print(f"[make_ai_move] Reward calculated: {reward}")
                    ai.update(((from_row, from_col), (to_row, to_col)), reward, board_before, board_after)
                    print(f"[make_ai_move] AI updated successfully")
                    # ادامه پرش‌های چندگانه
                    if self.multi_jump_active:
                        print(f"[make_ai_move] More jumps available, continuing for {ai_id}")
                        return self.make_ai_move(ai_id)
                    return True
                else:
                    print(f"[make_ai_move] Failed to move to ({to_row}, {to_col})")
                    self.valid_moves = {}  # پاک کردن valid_moves در صورت شکست
                    return False
            else:
                print(f"[make_ai_move] Failed to select piece at ({from_row}, {from_col})")
                self.valid_moves = {}  # پاک کردن valid_moves در صورت شکست
                return False

        except Exception as e:
            print(f"[make_ai_move] Error in AI move (AI ID: {ai_id}): {str(e)}")
            import traceback
            traceback.print_exc()
            self.valid_moves = {}  # پاک کردن valid_moves در صورت خطا
            return False

    def select(self, row, col):
        print(
            f"[Game.select] Selecting piece at ({row}, {col}), turn: {self.turn}, multi_jump_active: {self.multi_jump_active}")

        piece = self.board.board[row, col]
        print(f"[Game.select] Piece: {piece}")

        if piece != 0 and ((piece > 0 and not self.turn) or (piece < 0 and self.turn)):
            valid_moves = self.get_valid_moves(row, col)
            # تبدیل کلیدها به (end_row, end_col)
            self.valid_moves = {(move[2], move[3]): skipped for move, skipped in valid_moves.items()}
            print(f"[Game.select] Valid moves: {self.valid_moves}")

            if self.valid_moves:
                self.selected = (row, col)
                return True
        return False

    def _move(self, row, col):
        print(f"[Game._move] Attempting move to ({row}, {col}), selected: {self.selected}")

        if self.selected and (row, col) in self.valid_moves:
            start_row, start_col = self.selected
            piece = self.board.board[start_row, start_col]

            # جابه‌جایی مهره
            self.board.board[start_row, start_col] = 0
            self.board.board[row, col] = piece

            # حذف مهره‌های پریده‌شده
            skipped = self.valid_moves[(row, col)]
            for skip_row, skip_col in skipped:
                self.board.board[skip_row, skip_col] = 0

            # بررسی تبدیل به شاه
            if piece == 1 and row == 0 and not self.turn:
                self.board.board[row, col] = 2
                print(f"[Game._move] Promoted to king at ({row}, {col})")
            elif piece == -1 and row == 7 and self.turn:
                self.board.board[row, col] = -2
                print(f"[Game._move] Promoted to king at ({row}, {col})")

            # مدیریت پرش‌های چندگانه
            self.valid_moves = {}
            self.multi_jump_active = False
            if skipped:
                self.selected = (row, col)
                next_moves = self.get_valid_moves(row, col)
                next_jumps = {(move[2], move[3]): skipped for move, skipped in next_moves.items() if skipped}
                if next_jumps:
                    self.valid_moves = next_jumps
                    self.multi_jump_active = True
                    print(f"[Game._move] Multi-jump active, next jumps: {next_jumps}")
                else:
                    self.selected = None
                    self.turn = not self.turn
                    print(f"[Game._move] No more jumps, turn changed to {self.turn}")
            else:
                self.selected = None
                self.turn = not self.turn
                print(f"[Game._move] Simple move, turn changed to {self.turn}")

            # به‌روزرسانی تعداد مهره‌ها
            self.board.player_1_left = sum(1 for row in self.board.board.flat if row > 0)
            self.board.player_2_left = sum(1 for row in self.board.board.flat if row < 0)
            print(
                f"[Game._move] Board updated: player_1_left={self.board.player_1_left}, player_2_left={self.board.player_2_left}")
            return True

        print(f"[Game._move] Invalid move: selected={self.selected}, valid_moves={self.valid_moves}")
        return False

    def get_valid_moves(self, row, col):
        """
        محاسبه حرکت‌های مجاز برای مهره در موقعیت (row, col) با رعایت پرش اجباری.
        اگر هر مهره‌ای از بازیکن فعلی بتواند پرش کند، فقط پرش‌ها برگردانده می‌شوند.
        خروجی: دیکشنری {(start_row, start_col, end_row, end_col): [(skipped_row, skipped_col), ...]}
        """
        # print(f"[Game.get_valid_moves] Getting moves for ({row}, {col})")
        piece = self.board.board[row, col]
        if piece == 0:
            print(f"[Game.get_valid_moves] No piece at ({row}, {col})")
            return {}

        is_player_2 = piece < 0
        player_number = 2 if is_player_2 else 1
        is_king = abs(piece) == 2

        # جمع‌آوری تمام پرش‌های ممکن برای همه مهره‌های بازیکن
        all_jumps = {}
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.board.board[r, c]
                if p != 0 and ((p < 0 and player_number == 2) or (p > 0 and player_number == 1)):
                    jumps = self._get_jumps_recursive(r, c, p, player_number, abs(p) == 2, [], set())
                    for (end_row, end_col), skipped in jumps.items():
                        all_jumps[(r, c, end_row, end_col)] = skipped

        # اگر پرشی ممکن باشد، فقط پرش‌ها را برگردان
        if all_jumps:
            # print(f"[Game.get_valid_moves] Jumps available: {all_jumps}")
            return all_jumps

        # اگر هیچ پرشی ممکن نباشد، حرکت‌های ساده برای مهره در (row, col) بررسی می‌شوند
        moves = {}
        directions = self.get_piece_directions(piece, is_king)
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                target = self.board.board[new_row, new_col]
                if target == 0:
                    moves[(row, col, new_row, new_col)] = []

        # print(f"[Game.get_valid_moves] Moves: {moves}")
        return moves

    def _get_jumps_recursive(self, row, col, piece, player_number, is_king, skipped, visited):
        """
        محاسبه بازگشتی تمام پرش‌های ممکن برای مهره در (row, col).
        ورودی:
            row, col: موقعیت فعلی مهره
            piece: نوع مهره (مثبت برای بازیکن 1، منفی برای بازیکن 2)
            player_number: 1 یا 2
            is_king: آیا مهره شاه است
            skipped: لیست مهره‌های پرش‌شده در مسیر فعلی
            visited: مجموعه موقعیت‌های بازدیدشده برای جلوگیری از حلقه
        خروجی:
            دیکشنری پرش‌ها: {(jump_row, jump_col): [(skipped_row, skipped_col), ...]}
        """
        jumps = {}
        visited.add((row, col))

        # جهت‌های پرش
        directions = self.get_piece_directions(piece, is_king)

        # بررسی پرش‌های ممکن از موقعیت فعلی
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            jump_row, jump_col = new_row + dr, new_col + dc

            # بررسی شرایط پرش
            if (0 <= new_row < self.board_size and 0 <= new_col < self.board_size and
                    0 <= jump_row < self.board_size and 0 <= jump_col < self.board_size):
                target = self.board.board[new_row, new_col]
                jump_target = self.board.board[jump_row, jump_col]
                # پرش معتبر: هدف مهره حریف باشد و مقصد خالی
                if (target != 0 and ((target < 0) != (piece < 0)) and jump_target == 0):
                    new_skipped = skipped + [(new_row, new_col)]

                    # بررسی تبدیل به شاه
                    becomes_king = (not is_king and
                                    ((player_number == 1 and jump_row == 0) or
                                     (player_number == 2 and jump_row == self.board_size - 1)))

                    # ثبت پرش
                    jumps[(jump_row, jump_col)] = new_skipped

                    # اگر به شاه تبدیل نشد، بررسی پرش‌های بعدی
                    if not becomes_king:
                        # شبیه‌سازی پرش
                        original_piece = self.board.board[row, col]
                        original_target = self.board.board[new_row, new_col]
                        self.board.board[row, col] = 0
                        self.board.board[new_row, new_col] = 0
                        self.board.board[jump_row, jump_col] = piece

                        # بررسی پرش‌های بعدی
                        next_jumps = self._get_jumps_recursive(
                            jump_row, jump_col, piece, player_number, is_king,
                            new_skipped, visited.copy()
                        )

                        # بازگرداندن تخته
                        self.board.board[row, col] = original_piece
                        self.board.board[new_row, new_col] = original_target
                        self.board.board[jump_row, jump_col] = 0

                        # اضافه کردن پرش‌های بعدی
                        for (jr, jc), skip_list in next_jumps.items():
                            jumps[(jr, jc)] = skip_list

        return jumps

    def get_piece_directions(self, piece, is_king):
        """
        تعیین جهت‌های حرکت برای یک مهره.
        """
        is_player_2 = piece < 0
        player_number = 2 if is_player_2 else 1
        if not is_king and player_number == 1:
            return [(-1, -1), (-1, 1)]
        elif not is_king and player_number == 2:
            return [(1, -1), (1, 1)]
        elif is_king:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return []

    def _update_game(self):
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        if not self.game_started or self.game_over:
            return

        # بررسی حرکت‌های مجاز فقط برای بازیکن فعلی
        current_player_moves = False
        player_number = 2 if self.turn else 1  # بازیکن 1: turn=False, بازیکن 2: turn=True

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.board[row, col]
                # بررسی فقط برای مهره‌های بازیکن فعلی
                if piece != 0 and (
                        (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                ):
                    moves = self.get_valid_moves(row, col)
                    if moves:
                        current_player_moves = True
                        break  # اگر یک حرکت معتبر پیدا شد، نیازی به ادامه بررسی نیست

        # اگر بازیکن فعلی هیچ حرکتی نداشته باشد، حریف برنده است
        if not current_player_moves:
            self.winner = not self.turn  # حریف برنده است
            self.game_over = True
        elif self.no_capture_or_king_moves >= self.max_no_capture_moves:
            self.winner = None  # مساوی
            self.game_over = True

        # بررسی تایمر
        if self.settings.use_timer:
            player_1_time = self.timer.get_current_time(False)
            player_2_time = self.timer.get_current_time(True)
            if player_1_time <= 0:
                self.winner = True  # بازیکن 2 برنده است
                self.game_over = True
                self.time_up = True
            elif player_2_time <= 0:
                self.winner = False  # بازیکن 1 برنده است
                self.game_over = True
                self.time_up = True

        if self.game_over and not self.score_updated:
            self.score_updated = True
            if self.winner is not None:
                if not self.winner:
                    self.player_1_wins += 1
                else:
                    self.player_2_wins += 1
            save_stats({
                "player_1_wins": self.player_1_wins,
                "player_2_wins": self.player_2_wins
            })

    def handle_click(self, pos):
        x, y = pos
        self.click_log.append((x, y))
        if len(self.click_log) > 50:
            self.click_log.pop(0)

        if x < self.interface.BOARD_WIDTH and self.game_started and not self.game_over:
            is_human_turn = (
                    self.settings.game_mode == "human_vs_human" or
                    (self.settings.game_mode == "human_vs_ai" and not self.turn)
            )
            if is_human_turn:
                row = (y - self.interface.MENU_HEIGHT - self.interface.BORDER_THICKNESS) // self.interface.SQUARE_SIZE
                col = (x - self.interface.BORDER_THICKNESS) // self.interface.SQUARE_SIZE
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    # ذخیره حالت تخته قبل از حرکت
                    self.interface.move_history.append((
                        self.board.board.copy(),
                        self.turn,
                        self.valid_moves.copy(),
                        self.board.player_1_left,
                        self.board.player_2_left
                    ))
                    self.interface.redo_stack.clear()  # پاک کردن redo_stack پس از حرکت جدید
                    result = self._move(row, col)
                    return result
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    self.valid_moves = self.get_valid_moves(*self.selected)
                return result

        if self.game_over:
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
        return False

    def draw_valid_moves(self):
        if self.valid_moves:
            drawn_moves = set()
            for move in self.valid_moves:
                row, col = move
                if (row, col) not in drawn_moves:
                    # رسم دایره برای حرکت معتبر (مثلاً با Pygame)
                    print(f"[Game.draw_valid_moves] Drawing move at ({row}, {col})")
                    drawn_moves.add((row, col))

    def change_turn(self):
        self.selected = None
        self.valid_moves = {}
        self.turn = not self.turn
        print(f"[Game.change_turn] Turn changed to: {self.turn}")
        self.check_winner()  # بررسی برنده در ابتدای نوبت

    def get_hint(self):
        if self.game.game_over:
            return None
        # بررسی فعال بودن Hint برای بازیکن فعلی
        if (not self.game.turn and not self.hint_enabled_p1) or (self.game.turn and not self.hint_enabled_p2):
            return None
        ai_id = "ai_1" if not self.game.turn else "ai_2"
        ai = self.ai_players.get(ai_id, None)
        if ai and (self.settings.game_mode in ["human_vs_human", "human_vs_ai"]):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    piece = self.game.board.board[row, col]
                    if piece != 0 and (piece < 0) == self.game.turn:
                        moves = self.game.get_valid_moves(row, col)
                        if moves:
                            try:
                                move = ai.suggest_move(self.game, moves) if hasattr(ai, 'suggest_move') else ai.act(
                                    moves)
                                if move:
                                    return ((row, col), move)
                            except Exception as e:
                                print(f"Error in AI hint (AI ID: {ai.ai_id}): {e}")
        return None

    def save_game_state(self):
        """ذخیره وضعیت فعلی بازی"""
        from copy import deepcopy
        state = {
            'board': deepcopy(self.game.board.board),
            'turn': self.game.turn,
            'player_1_left': self.game.board.player_1_left,
            'player_2_left': self.game.board.player_2_left,
            'timer': {
                'player_1_time': self.game.timer.get_current_time(False),
                'player_2_time': self.game.timer.get_current_time(True)
            }
        }
        self.move_history.append(state)
        self.redo_stack.clear()  # با حرکت جدید، Redo غیرفعال می‌شود
