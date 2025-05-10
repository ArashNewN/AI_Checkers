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
        print("Game.__init__ called")  # لاگ دیباگ
        config = load_config()
        self.board_size = config.get("network_params", {}).get("board_size", 8)
        self.max_no_capture_moves = config.get("max_no_capture_moves", 40)

        # تنظیمات و واسط گرافیکی
        self.settings = settings
        self.interface = interface

        # صفحه نمایش
        self.screen = pygame.display.set_mode((config.get("window_width", 940), config.get("window_height", 727)))
        pygame.display.set_caption('Checkers: Player 1 vs Player 2')

        # فونت‌ها
        self.font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 24)
        self.small_font = pygame.font.SysFont('Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial', 18)

        # آمار و وضعیت بازیکنان
        stats = load_stats()
        self.player_1_wins = stats["player_1_wins"]
        self.player_2_wins = stats["player_2_wins"]

        # مدیریت AIها
        self.ai_players = {}
        if self.settings.game_mode in ["human_vs_ai", "ai_vs_ai"] or \
                self.settings.player_1_ai_type != "none" or \
                self.settings.player_2_ai_type != "none":
            self.reward_calculator = RewardCalculator(self)
        else:
            self.reward_calculator = None
        self.update_ai_players()

        # تایمر
        self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)

        # وضعیت بازی
        self.multi_jump_active = False
        self.no_capture_or_king_moves = 0
        self.last_update_time = pygame.time.get_ticks()

        # شروع اولیه
        self.init_game()

    def update_ai_players(self):
        print(f"Updating AI players. Current ai_players: {self.ai_players}")
        config_dict = load_ai_config()
        #print(f"Config dict in update_ai_players: {config_dict}")

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

        self.ai_players = {}  # ریست کردن ai_players
        if self.settings.game_mode == "ai_vs_ai":
            players = ["player_1", "player_2"]
        elif self.settings.game_mode == "human_vs_ai":
            players = ["player_2"]  # فقط برای ai_2 در human_vs_ai
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
                        print(f"AI players after update: {self.ai_players}")
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
            print(
                f"Config for {ai_id}: {config_dict['ai_configs']['player_1' if ai_id == 'ai_1' else 'player_2']}")  # لاگ دیباگ
            return ai_class(game=self, model_name=model_name, ai_id=ai_id, config=config_dict)
        except Exception as e:
            print(f"Error creating AI {ai_id}: {e}")
            return None

    def init_game(self):
        """راه‌اندازی اولیه بازی."""
        print("init_game called")  # لاگ دیباگ
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
        """شروع بازی."""
        print("start_game called")  # لاگ دیباگ
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
        """ریست کردن صفحه بازی."""
        print("reset_board called")  # لاگ دیباگ
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
                valid_moves = ai.get_valid_moves(self.board.board)
                print(f"[make_ai_move] Valid moves from AI: {valid_moves}")
                move = ai.get_move(self.board.board)
                print(f"[make_ai_move] AI move returned: {move}")
                if move:
                    (from_row, from_col), (to_row, to_col) = move
                    success = self.select(from_row, from_col)
                    if success:
                        self._move(to_row, to_col)
                        board_after = self.board.board.copy()
                        reward = self.reward_calculator.get_reward(player_number=ai.player_number)
                        print(f"[make_ai_move] Reward calculated: {reward}")
                        ai.update(move, reward, board_before, board_after)
                        print(f"[make_ai_move] AI updated successfully")
                        return True
                    else:
                        print(f"[make_ai_move] Failed to select piece at ({from_row}, {from_col})")
                        return False
                else:
                    print("[make_ai_move] Warning: AI returned None for move")
                    return False
            except Exception as e:
                print(f"[make_ai_move] Error in AI move (AI ID: {ai_id}): {str(e)}")
                import traceback
                traceback.print_exc()
                return False

    def select(self, row, col):
        """انتخاب مهره."""
        if self.game_over or not self.game_started:
            return False
        if self.selected:
            result = self._move(row, col)
            if not result and not self.multi_jump_active:
                self.selected = None
                return self.select(row, col)
            return result

        piece = self.board.board[row, col]
        # استفاده از player_number به جای علامت مهره
        player_number = 1 if not self.turn else 2
        if piece != 0 and (
                (piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)
        ):
            if self.multi_jump_active and (row, col) != self.selected:
                return False
            self.selected = (row, col)
            self.valid_moves = self.get_valid_moves(row, col)
            return True
        return False

    def get_valid_moves(self, row, col):
        """دریافت حرکت‌های معتبر برای یک مهره."""
        moves = {}
        jumps = {}
        piece = self.board.board[row, col]
        if piece == 0:
            return moves

        # تعیین بازیکن بر اساس علامت مهره
        is_player_2 = piece < 0
        player_number = 2 if is_player_2 else 1
        is_king = abs(piece) == 2
        directions = []
        if not is_king and player_number == 1:
            directions = [(-1, -1), (-1, 1)]
        elif not is_king and player_number == 2:
            directions = [(1, -1), (1, 1)]
        elif is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        has_jump = False
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.board.board[r, c]
                if p != 0 and ((p < 0 and player_number == 2) or (p > 0 and player_number == 1)):
                    piece_directions = self.get_piece_directions(p, abs(p) == 2)
                    for dr, dc in piece_directions:
                        new_row, new_col = r + dr, c + dc
                        if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                            target = self.board.board[new_row, new_col]
                            if target != 0 and ((target < 0) != (p < 0)):
                                jump_row, jump_col = new_row + dr, new_col + dc
                                if (0 <= jump_row < self.board_size and 0 <= jump_col < self.board_size and
                                        self.board.board[jump_row, jump_col] == 0):
                                    has_jump = True
                                    if r == row and c == col:
                                        jumps[(jump_row, jump_col)] = [(new_row, new_col)]
        if has_jump:
            return jumps if jumps else {}

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size:
                target = self.board.board[new_row, new_col]
                if target == 0:
                    moves[(new_row, new_col)] = []
        return moves

    def _update_game(self):
        """به‌روزرسانی وضعیت بازی."""
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        """بررسی برنده بازی."""
        if not self.game_started or self.game_over:
            return

        player_1_moves = False
        player_2_moves = False

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.board[row, col]
                if piece != 0:
                    moves = self.get_valid_moves(row, col)
                    if moves:
                        if piece > 0:
                            player_1_moves = True
                        else:
                            player_2_moves = True

        if not player_1_moves:
            self.winner = True
            self.game_over = True
        elif not player_2_moves:
            self.winner = False
            self.game_over = True
        elif self.no_capture_or_king_moves >= self.max_no_capture_moves:
            self.winner = None
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
        """مدیریت کلیک کاربر."""
        x, y = pos
        self.click_log.append((x, y))
        if len(self.click_log) > 50:
            self.click_log.pop(0)

        if x < BOARD_WIDTH and self.game_started and not self.game_over:
            is_human_turn = (
                self.settings.game_mode == "human_vs_human" or
                (self.settings.game_mode == "human_vs_ai" and not self.turn)
            )
            if is_human_turn:
                row = (y - MENU_HEIGHT - BORDER_THICKNESS) // SQUARE_SIZE
                col = (x - BORDER_THICKNESS) // SQUARE_SIZE
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    return self._move(row, col)
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    self.valid_moves = self.get_valid_moves(*self.selected)
                return result

        if self.game_over:
            self.selected = None
            self.valid_moves = {}
            self.multi_jump_active = False
        return False

    def _move(self, row, col):
        """اجرای حرکت."""
        if not self.selected or (row, col) not in self.valid_moves:
            return False

        start_row, start_col = self.selected
        piece = self.board.board[start_row, start_col]
        if piece != 0:
            self.board.board[start_row, start_col] = 0
            skipped = self.valid_moves[(row, col)]

            if skipped:
                self.board.remove([(r, c) for r, c in skipped])
                self.no_capture_or_king_moves = 0
            else:
                self.no_capture_or_king_moves += 1

            self.board.board[row, col] = piece
            self.move_log.append((start_row, start_col, row, col))
            if len(self.move_log) > 30:
                self.move_log.pop(0)

            kinged = False
            if piece > 0 and row == 0 and abs(piece) != 2:
                self.board.board[row, col] = 2
                kinged = True
                self.no_capture_or_king_moves = 0
            elif piece < 0 and row == self.board_size - 1 and abs(piece) != 2:
                self.board.board[row, col] = -2
                kinged = True
                self.no_capture_or_king_moves = 0

            if self.interface:
                self.interface.animate_move(None, start_row, start_col, row, col)

            if kinged:
                self.multi_jump_active = False
                self.change_turn()
                return True

            if skipped:
                additional_moves = self.get_valid_moves(row, col)
                if any(skipped for skipped in additional_moves.values()):
                    self.valid_moves = {move: skip for move, skip in additional_moves.items() if skip}
                    self.multi_jump_active = True
                    self.selected = (row, col)
                    return True

            self.multi_jump_active = False
            self.change_turn()
            return True
        return False

    def get_piece_directions(self, piece, is_king):
        """دریافت جهت‌های حرکت برای یک مهره."""
        if not is_king and piece > 0:
            return [(-1, -1), (-1, 1)]
        elif not is_king and piece < 0:
            return [(1, -1), (1, 1)]
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def draw_valid_moves(self):
        """نمایش حرکت‌های معتبر."""
        if self.valid_moves:
            for move in self.valid_moves:
                row, col = move
                pygame.draw.circle(self.screen, RED,
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2 + BORDER_THICKNESS,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2 + MENU_HEIGHT + BORDER_THICKNESS), 20)

    def change_turn(self):
        """تغییر نوبت."""
        self.selected = None
        self.valid_moves = {}
        self.turn = not self.turn

    def get_hint(self):
        """دریافت پیشنهاد حرکت از AI."""
        if self.game_over:
            return None
        # انتخاب ai_id بر اساس نوبت
        ai_id = "ai_1" if not self.turn else "ai_2"
        ai = self.ai_players.get(ai_id)
        if ai:
            for row in range(self.board_size):
                for col in range(self.board_size):
                    piece = self.board.board[row, col]
                    if piece != 0 and (piece < 0) == self.turn:
                        moves = self.get_valid_moves(row, col)
                        if moves:
                            try:
                                move = ai.suggest_move(self, moves) if hasattr(ai, 'suggest_move') else ai.act(self,
                                                                                                               moves)
                                if move:
                                    return ((row, col), move)
                            except Exception as e:
                                print(f"Error in AI hint (AI ID: {ai.ai_id}): {e}")
        return None

