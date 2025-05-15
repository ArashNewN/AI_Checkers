import pygame
import numpy as np
import importlib
from .board import Board
from .timer import TimerManager
from .rewards import RewardCalculator
from .config import load_config, load_stats, save_stats, load_ai_config
from copy import deepcopy


class MoveHistory:
    """مدیریت تاریخچه حرکات و پشته redo در بازی."""

    def __init__(self):
        self.move_history = []
        self.redo_stack = []

    def save_state(self, board, turn, valid_moves, player_1_left, player_2_left):
        """
        ذخیره وضعیت فعلی بازی در تاریخچه.

        Args:
            board (numpy.ndarray): آرایه تخته بازی
            turn (bool): نوبت فعلی (False برای بازیکن 1، True برای بازیکن 2)
            valid_moves (dict): حرکت‌های معتبر
            player_1_left (int): تعداد مهره‌های باقی‌مانده بازیکن 1
            player_2_left (int): تعداد مهره‌های باقی‌مانده بازیکن 2
        """
        state = {
            'board': deepcopy(board),
            'turn': turn,
            'valid_moves': deepcopy(valid_moves),
            'player_1_left': player_1_left,
            'player_2_left': player_2_left
        }
        self.move_history.append(state)
        self.redo_stack.clear()

    def clear_redo_stack(self):
        """پاک کردن پشته redo."""
        self.redo_stack.clear()

    def save_game_state(self, game):
        """
        ذخیره وضعیت کامل بازی شامل تایمر.

        Args:
            game (Game): نمونه کلاس Game
        """
        state = {
            'board': deepcopy(game.board.board),
            'turn': game.turn,
            'player_1_left': game.board.player_1_left,
            'player_2_left': game.board.player_2_left,
            'timer': {
                'player_1_time': game.timer.get_current_time(False),
                'player_2_time': game.timer.get_current_time(True)
            }
        }
        self.move_history.append(state)
        self.redo_stack.clear()


class Game:
    """کلاس اصلی بازی شطرنج چینی (Checkers) با پشتیبانی از حالت‌های مختلف بازی."""

    def __init__(self, settings, interface=None):
        """مقداردهی اولیه بازی.

        Args:
            settings: تنظیمات بازی (مثل game_mode و player_starts)
            interface: رابط کاربری (اختیاری)
        """
        config = load_config()
        self.board_size = config.get("board_size", 8)
        self.max_no_capture_moves = config.get("max_no_capture_moves", 40)
        self.max_uniform_moves = config.get("max_uniform_moves", 5)
        self.max_total_moves = config.get("max_total_moves", 40)
        self.window_width = config.get("window_width", 940)
        self.window_height = config.get("window_height", 720)
        self.board_width = config.get("board_width", 640)
        self.menu_height = config.get("menu_height", 80)
        self.border_thickness = config.get("border_thickness", 10)
        self.square_size = config.get("square_size", 80)
        self.red = config.get("colors", {}).get("red", [255, 0, 0])
        self.hint_enabled_p1 = config.get("hint_enabled_p1", True)
        self.hint_enabled_p2 = config.get("hint_enabled_p2", True)

        self.settings = settings
        self.interface = interface
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Checkers: Player 1 vs Player 2')

        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
        self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)

        stats = load_stats()
        self.player_1_wins = stats["player_1_wins"]
        self.player_2_wins = stats["player_2_wins"]

        self.ai_players = {}
        if (self.settings.game_mode in ["human_vs_ai", "ai_vs_ai"] or
                self.settings.player_1_ai_type != "none" or
                self.settings.player_2_ai_type != "none"):
            self.reward_calculator = RewardCalculator(self)
        else:
            self.reward_calculator = None
        self.update_ai_players()

        self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)
        self.history = MoveHistory()
        self.consecutive_uniform_moves = {1: 0, 2: 0}
        self.total_moves = {1: 0, 2: 0}
        self.multi_jump_active = False
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()
        self.board = None
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

        self.init_game()

    def update_ai_players(self):
        """به‌روزرسانی بازیکنان هوش مصنوعی بر اساس تنظیمات."""
        print(f"Updating AI players. Current ai_players: {self.ai_players}")
        config_dict = load_ai_config()

        ai_type_to_class = {}
        for ai_info in config_dict.get("available_ais", []):
            if "module" not in ai_info or "class" not in ai_info:
                print(f"Error: Missing 'module' or 'class' in ai_info: {ai_info}")
                continue
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
        """ایجاد یک نمونه AI با نوع و مدل مشخص.

        Args:
            ai_type (str): نوع AI
            model_name (str): نام مدل AI
            ai_id (str): شناسه AI ('ai_1' یا 'ai_2')

        Returns:
            object: نمونه AI یا None در صورت خطا
        """
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
        """مقداردهی اولیه وضعیت بازی."""
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
        self.consecutive_uniform_moves = {1: 0, 2: 0}
        self.total_moves = {1: 0, 2: 0}
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.repeat_count = 0

    def start_game(self):
        """شروع یک بازی جدید."""
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
        """ریست کردن تخته بازی."""
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
        self.consecutive_uniform_moves = {1: 0, 2: 0}
        self.total_moves = {1: 0, 2: 0}
        self.timer.start_game()
        self.last_update_time = pygame.time.get_ticks()
        self.update_ai_players()

    def make_ai_move(self, ai_id=None):
        """اجرای حرکت توسط AI.

        Args:
            ai_id (str, optional): شناسه AI ('ai_1' یا 'ai_2'). اگر None باشد، بر اساس نوبت انتخاب می‌شود.

        Returns:
            bool: True اگر حرکت موفق بود، False در غیر این صورت
        """
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

            player_number = 1 if ai_id == "ai_1" else 2
            print(f"[make_ai_move] Player number: {player_number}")

            if self.multi_jump_active and self.selected:
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

            from_row, from_col, to_row, to_col = move
            print(f"[make_ai_move] Attempting move from ({from_row}, {from_col}) to ({to_row}, {to_col})")

            if self.multi_jump_active and self.selected and (from_row, from_col) != self.selected:
                print(f"[make_ai_move] Resetting self.selected to ({from_row}, {from_col})")
                self.selected = (from_row, from_col)

            self.history.save_state(
                self.board.board,
                self.turn,
                valid_moves,
                self.board.player_1_left,
                self.board.player_2_left
            )

            print(f"[make_ai_move] Selecting piece at ({from_row}, {from_col})")
            success = self.select(from_row, from_col)
            if success:
                print(f"[make_ai_move] Selection successful, moving to ({to_row}, {to_col})")
                success = self._move(to_row, to_col)
                if success:
                    board_after = self.board.board.copy()
                    reward = self.reward_calculator.get_reward(player_number=player_number)
                    print(f"[make_ai_move] Reward calculated: {reward}")
                    ai.update(((from_row, from_col), (to_row, to_col)), reward, board_before, board_after)
                    print(f"[make_ai_move] AI updated successfully")
                    if self.multi_jump_active:
                        print(f"[make_ai_move] More jumps available, continuing for {ai_id}")
                        return self.make_ai_move(ai_id)
                    return True
                else:
                    print(f"[make_ai_move] Failed to move to ({to_row}, {to_col})")
                    self.valid_moves = {}
                    return False
            else:
                print(f"[make_ai_move] Failed to select piece at ({from_row}, {from_col})")
                self.valid_moves = {}
                return False

        except Exception as e:
            print(f"[make_ai_move] Error in AI move (AI ID: {ai_id}): {str(e)}")
            import traceback
            traceback.print_exc()
            self.valid_moves = {}
            return False

    def select(self, row, col):
        """انتخاب یک مهره در موقعیت (row, col).

        Args:
            row (int): شماره ردیف
            col (int): شماره ستون

        Returns:
            bool: True اگر انتخاب موفق بود، False در غیر این صورت
        """
        print(
            f"[Game.select] Selecting piece at ({row}, {col}), turn: {self.turn}, multi_jump_active: {self.multi_jump_active}")

        piece = self.board.board[row, col]
        print(f"[Game.select] Piece: {piece}")

        if piece != 0 and ((piece > 0 and not self.turn) or (piece < 0 and self.turn)):
            valid_moves = self.get_valid_moves(row, col)
            self.valid_moves = {(move[2], move[3]): skipped for move, skipped in valid_moves.items()}
            print(f"[Game.select] Valid moves: {self.valid_moves}")

            if self.valid_moves:
                self.selected = (row, col)
                return True
        return False

    def _move(self, row, col):
        """انجام حرکت از مهره انتخاب‌شده به موقعیت (row, col).

        Args:
            row (int): شماره ردیف مقصد
            col (int): شماره ستون مقصد

        Returns:
            bool: True اگر حرکت موفق بود، False در غیر این صورت
        """
        print(f"[Game._move] Attempting move to ({row}, {col}), selected: {self.selected}")
        if self.selected and (row, col) in self.valid_moves:
            start_row, start_col = self.selected
            piece = self.board.board[start_row, start_col]
            player_number = 2 if piece < 0 else 1

            self.board.board[start_row, start_col] = 0
            self.board.board[row, col] = piece

            skipped = self.valid_moves[(row, col)]
            is_capture = bool(skipped)

            for skip_row, skip_col in skipped:
                self.board.board[skip_row, skip_col] = 0

            is_promotion = False
            if piece == 1 and row == 0 and not self.turn:
                self.board.board[row, col] = 2
                is_promotion = True
                print(f"[Game._move] Promoted to king at ({row}, {col})")
            elif piece == -1 and row == 7 and self.turn:
                self.board.board[row, col] = -2
                is_promotion = True
                print(f"[Game._move] Promoted to king at ({row}, {col})")

            self.total_moves[player_number] += 1
            if not (is_capture or is_promotion):
                self.consecutive_uniform_moves[player_number] += 1
            else:
                self.consecutive_uniform_moves[player_number] = 0
            if is_capture or is_promotion:
                self.no_capture_or_king_moves = 0
            else:
                self.no_capture_or_king_moves += 1

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

            self.board.player_1_left = sum(1 for row in self.board.board.flat if row > 0)
            self.board.player_2_left = sum(1 for row in self.board.board.flat if row < 0)
            print(
                f"[Game._move] Board updated: player_1_left={self.board.player_1_left}, player_2_left={self.board.player_2_left}")
            return True

        print(f"[Game._move] Invalid move: selected={self.selected}, valid_moves={self.valid_moves}")
        return False

    def get_valid_moves(self, row, col):
        """محاسبه حرکت‌های مجاز برای مهره در موقعیت (row, col).

        Args:
            row (int): شماره ردیف مهره
            col (int): شماره ستون مهره

        Returns:
            dict: دیکشنری حرکت‌های معتبر با کلید (start_row, start_col, end_row, end_col)
                  و مقدار لیست مهره‌های پریده‌شده
        """
        print(f"[Game.get_valid_moves] Getting moves for ({row}, {col})")
        piece = self.board.board[row, col]
        if piece == 0:
            print(f"[Game.get_valid_moves] No piece at ({row}, {col})")
            return {}

        is_player_2 = piece < 0
        player_number = 2 if is_player_2 else 1
        is_king = abs(piece) == 2

        all_jumps = {}
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.board.board[r, c]
                if p != 0 and ((p < 0 and player_number == 2) or (p > 0 and player_number == 1)):
                    jumps = self._get_jumps_recursive(r, c, p, player_number, abs(p) == 2, [], set())
                    for (end_row, end_col), skipped in jumps.items():
                        all_jumps[(r, c, end_row, end_col)] = skipped

        if all_jumps:
            print(f"[Game.get_valid_moves] Jumps available: {all_jumps}")
            return all_jumps

        moves = {}
        directions = self.get_piece_directions(piece, is_king)
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                target = self.board.board[new_row, new_col]
                if target == 0:
                    moves[(row, col, new_row, new_col)] = []

        print(f"[Game.get_valid_moves] Moves: {moves}")
        return moves

    def _get_jumps_recursive(self, row, col, piece, player_number, is_king, skipped, visited):
        """محاسبه بازگشتی پرش‌های ممکن برای مهره.

        Args:
            row (int): شماره ردیف فعلی
            col (int): شماره ستون فعلی
            piece (int): نوع مهره
            player_number (int): شماره بازیکن (1 یا 2)
            is_king (bool): آیا مهره شاه است
            skipped (list): لیست مهره‌های پریده‌شده
            visited (set): مجموعه موقعیت‌های بازدیدشده

        Returns:
            dict: دیکشنری پرش‌ها با کلید (jump_row, jump_col) و مقدار لیست مهره‌های پریده‌شده
        """
        jumps = {}
        visited.add((row, col))

        directions = self.get_piece_directions(piece, is_king)

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            jump_row, jump_col = new_row + dr, new_col + dc

            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size and \
                    0 <= jump_row < self.board_size and 0 <= jump_col < self.board_size:
                target = self.board.board[new_row, new_col]
                jump_target = self.board.board[jump_row, jump_col]
                if target != 0 and ((target < 0) != (piece < 0)) and jump_target == 0:
                    new_skipped = skipped + [(new_row, new_col)]
                    becomes_king = (not is_king and
                                    ((player_number == 1 and jump_row == 0) or
                                     (player_number == 2 and jump_row == self.board_size - 1)))
                    jumps[(jump_row, jump_col)] = new_skipped
                    if not becomes_king:
                        original_piece = self.board.board[row, col]
                        original_target = self.board.board[new_row, new_col]
                        self.board.board[row, col] = 0
                        self.board.board[new_row, new_col] = 0
                        self.board.board[jump_row, jump_col] = piece
                        next_jumps = self._get_jumps_recursive(
                            jump_row, jump_col, piece, player_number, is_king,
                            new_skipped, visited.copy()
                        )
                        self.board.board[row, col] = original_piece
                        self.board.board[new_row, new_col] = original_target
                        self.board.board[jump_row, jump_col] = 0
                        for (jr, jc), skip_list in next_jumps.items():
                            jumps[(jr, jc)] = skip_list

        return jumps

    @staticmethod
    def get_piece_directions(piece, is_king):
        """تعیین جهت‌های حرکت برای یک مهره.

        Args:
            piece (int): نوع مهره (مثبت برای بازیکن 1، منفی برای بازیکن 2)
            is_king (bool): آیا مهره شاه است

        Returns:
            list: لیست جهت‌های حرکت (dr, dc)
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

    def update_game(self):
        """به‌روزرسانی عمومی وضعیت بازی."""
        self._update_game()

    def _update_game(self):
        """به‌روزرسانی داخلی وضعیت بازی."""
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        """بررسی برنده یا تساوی بازی."""
        if not self.game_started or self.game_over:
            return

        current_player_moves = False
        player_number = 2 if self.turn else 1

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.board[row, col]
                if piece != 0 and (
                        (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
                ):
                    moves = self.get_valid_moves(row, col)
                    if moves:
                        current_player_moves = True
                        break

        if self.consecutive_uniform_moves[1] >= self.max_uniform_moves and \
                self.consecutive_uniform_moves[2] >= self.max_uniform_moves:
            self.winner = None
            self.game_over = True
        elif self.total_moves[1] >= self.max_total_moves and self.total_moves[2] >= self.max_total_moves:
            self.winner = None
            self.game_over = True
        elif self.no_capture_or_king_moves >= self.max_no_capture_moves:
            self.winner = None
            self.game_over = True
        elif not current_player_moves:
            self.winner = not self.turn
            self.game_over = True

        if self.settings.use_timer:
            player_1_time = self.timer.get_current_time(False)
            player_2_time = self.timer.get_current_time(True)
            if player_1_time <= 0:
                self.winner = True
                self.game_over = True
                self.time_up = True
            elif player_2_time <= 0:
                self.winner = False
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
        """مدیریت کلیک کاربر روی تخته.

        Args:
            pos (tuple): مختصات کلیک (x, y)

        Returns:
            bool: True اگر حرکت یا انتخاب موفق بود، False در غیر این صورت
        """
        x, y = pos
        self.click_log.append((x, y))
        if len(self.click_log) > 50:
            self.click_log.pop(0)

        if x < self.board_width and self.game_started and not self.game_over:
            is_human_turn = (
                    self.settings.game_mode == "human_vs_human" or
                    (self.settings.game_mode == "human_vs_ai" and not self.turn)
            )
            if is_human_turn:
                row = (y - self.menu_height - self.border_thickness) // self.square_size
                col = (x - self.border_thickness) // self.square_size
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    self.history.save_state(
                        self.board.board,
                        self.turn,
                        self.valid_moves,
                        self.board.player_1_left,
                        self.board.player_2_left
                    )
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
        """نمایش حرکت‌های معتبر روی تخته."""
        if self.valid_moves:
            drawn_moves = set()
            for move in self.valid_moves:
                row, col = move
                if (row, col) not in drawn_moves:
                    print(f"[Game.draw_valid_moves] Drawing move at ({row}, {col})")
                    drawn_moves.add((row, col))

    def change_turn(self):
        """تغییر نوبت بازیکن."""
        self.selected = None
        self.valid_moves = {}
        self.turn = not self.turn
        print(f"[Game.change_turn] Turn changed to: {self.turn}")
        self.check_winner()

    def get_hint(self):
        """دریافت پیشنهاد حرکت از AI برای بازیکن فعلی.

        Returns:
            tuple: ((from_row, from_col), (to_row, to_col)) یا None
        """
        if self.game_over:
            return None
        if not self.turn and not self.hint_enabled_p1 or self.turn and not self.hint_enabled_p2:
            return None
        ai_id = "ai_1" if not self.turn else "ai_2"
        ai = self.ai_players.get(ai_id, None)
        if ai and (self.settings.game_mode in ["human_vs_human", "human_vs_ai"]):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    piece = self.board.board[row, col]
                    if piece != 0 and (piece < 0) == self.turn:
                        moves = self.get_valid_moves(row, col)
                        if moves:
                            try:
                                move = None
                                if hasattr(ai, 'suggest_move'):
                                    move = ai.suggest_move(self, moves)
                                elif hasattr(ai, 'get_move'):
                                    move = ai.get_move(self.board.board)
                                if move:
                                    from_row, from_col, to_row, to_col = move
                                    return ((row, col), (to_row, to_col))
                            except Exception as e:
                                print(f"Error in AI hint (AI ID: {ai_id}): {e}")
        return None

    def save_game_state(self):
        """ذخیره وضعیت فعلی بازی."""
        self.history.save_game_state(self)