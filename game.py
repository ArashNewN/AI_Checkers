import pygame
import importlib
import logging
from copy import deepcopy
from .board import Board
from .timer import TimerManager
from .rewards import RewardCalculator
from .config import load_config, load_stats, save_stats, load_ai_config

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MoveHistory:
    def __init__(self):
        self.move_history = []
        self.redo_stack = []

    def save_state(self, game):
        """ذخیره وضعیت بازی."""
        try:
            formatted_valid_moves = {}
            for move, skipped in game.valid_moves.items():
                if len(move) == 4:
                    end_row, end_col = move[2], move[3]
                elif len(move) == 2:
                    end_row, end_col = move
                else:
                    logger.error(f"Invalid move format: {move}")
                    continue
                formatted_valid_moves[(end_row, end_col)] = skipped

            state = {
                'board': deepcopy(game.board.board),
                'turn': game.turn,
                'valid_moves': formatted_valid_moves,
                'player_1_left': game.board.player_1_left,
                'player_2_left': game.board.player_2_left,
                'multi_jump_active': game.multi_jump_active,
                'selected': game.selected,
                'timer': {
                    'player_1_time': game.timer.get_current_time(False),
                    'player_2_time': game.timer.get_current_time(True)
                }
            }
            self.move_history.append(state)
            self.redo_stack.clear()
            logger.debug("Game state saved")
        except Exception as e:
            logger.error(f"Error saving game state: {e}")

    def undo(self, game):
        """بازگرداندن حرکت قبلی."""
        try:
            if not self.move_history:
                logger.info("No moves to undo")
                return

            current_state = {
                'board': deepcopy(game.board.board),
                'turn': game.turn,
                'player_1_left': game.board.player_1_left,
                'player_2_left': game.board.player_2_left,
                'multi_jump_active': game.multi_jump_active,
                'selected': game.selected,
                'valid_moves': deepcopy(game.valid_moves),
                'timer': {
                    'player_1_time': game.timer.get_current_time(False),
                    'player_2_time': game.timer.get_current_time(True)
                }
            }
            self.redo_stack.append(current_state)

            last_state = self.move_history.pop()
            game.board.board = last_state['board']
            game.turn = last_state['turn']
            game.board.player_1_left = last_state['player_1_left']
            game.board.player_2_left = last_state['player_2_left']
            game.multi_jump_active = last_state.get('multi_jump_active', False)
            game.selected = last_state.get('selected', None)
            game.valid_moves = last_state.get('valid_moves', {})
            if 'timer' in last_state:
                game.timer.set_time(False, last_state['timer']['player_1_time'])
                game.timer.set_time(True, last_state['timer']['player_2_time'])

            logger.info(f"Restored state: turn={game.turn}, multi_jump_active={game.multi_jump_active}")
        except Exception as e:
            logger.error(f"Error undoing move: {e}")

class Game:
    """کلاس اصلی بازی شطرنج چینی (Checkers)."""

    def __init__(self, settings, interface=None):
        """مقداردهی اولیه بازی."""
        try:
            config = load_config()
            self._load_config(config)
            self.settings = settings
            self.interface = interface
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption('Checkers: Player 1 vs Player 2')
            self._init_fonts()
            self._init_stats()
            self._init_ai_players()
            self.timer = TimerManager(self.settings.use_timer, self.settings.game_time)
            self.history = MoveHistory()
            self._reset_game_state()
            self.init_game()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")
            raise

    def _load_config(self, config):
        """بارگذاری تنظیمات از config.json."""
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

    def _init_fonts(self):
        """مقداردهی اولیه فونت‌ها."""
        font_name = 'Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial'
        self.font = pygame.font.SysFont(font_name, 24)
        self.small_font = pygame.font.SysFont(font_name, 18)

    def _init_stats(self):
        """بارگذاری آمار بازی."""
        stats = load_stats()
        self.player_1_wins = stats.get("player_1_wins", 0)
        self.player_2_wins = stats.get("player_2_wins", 0)

    def _init_ai_players(self):
        """مقداردهی اولیه بازیکنان AI."""
        self.ai_players = {}
        if self.settings.game_mode in ["human_vs_ai", "ai_vs_ai"] or self.settings.player_1_ai_type != "none" or self.settings.player_2_ai_type != "none":
            self.reward_calculator = RewardCalculator(self)
        else:
            self.reward_calculator = None
        self.update_ai_players()

    def _reset_game_state(self):
        """ریست کردن وضعیت بازی."""
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
        self.paused = False

    def update_ai_players(self):
        """به‌روزرسانی بازیکنان هوش مصنوعی."""
        try:
            config_dict = load_ai_config()
            ai_type_to_class = self._load_ai_classes(config_dict)
            if not ai_type_to_class:
                logger.error("No valid AI classes loaded")
                return

            players = []
            if self.settings.game_mode == "ai_vs_ai":
                players = ["player_1", "player_2"]
            elif self.settings.game_mode == "human_vs_ai":
                players = ["player_2"]

            for player in players:
                ai_id = "ai_1" if player == "player_1" else "ai_2"
                settings = config_dict["ai_configs"].get(player, {})
                ai_type = settings.get("ai_type")
                model_name = settings.get("ai_code", "default")
                if ai_type and ai_type != "none":
                    ai_class = ai_type_to_class.get(ai_type)
                    if ai_class:
                        self.ai_players[ai_id] = ai_class(game=self, model_name=model_name, ai_id=ai_id, config=config_dict)
                        logger.info(f"Loaded AI: {ai_type} for {player} (ID: {ai_id})")
                    else:
                        logger.error(f"No AI class found for type: {ai_type}")
        except Exception as e:
            logger.error(f"Error updating AI players: {e}")

    def _load_ai_classes(self, config_dict):
        """بارگذاری کلاس‌های AI از تنظیمات."""
        ai_type_to_class = {}
        for ai_info in config_dict.get("available_ais", []):
            try:
                module_name = ai_info.get("module")
                class_name = ai_info.get("class")
                ai_type = ai_info.get("type")
                if not module_name or not class_name:
                    logger.error(f"Missing module or class in ai_info: {ai_info}")
                    continue
                module = importlib.import_module(module_name)
                ai_class = getattr(module, class_name)
                if not isinstance(ai_class, type):
                    logger.error(f"{class_name} is not a class")
                    continue
                ai_type_to_class[ai_type] = ai_class
                logger.debug(f"Mapped {ai_type} to {class_name}")
            except Exception as e:
                logger.error(f"Error importing AI class {class_name}: {e}")
        return ai_type_to_class

    def init_game(self):
        """مقداردهی اولیه وضعیت بازی."""
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.timer.start_game()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")

    def start_game(self):
        """شروع یک بازی جدید."""
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
        """ریست کردن تخته بازی."""
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.timer.start_game()
            self.update_ai_players()
            logger.info("Board reset")
        except Exception as e:
            logger.error(f"Error resetting board: {e}")

    def make_ai_move(self, ai_id=None):
        """اجرای حرکت توسط AI."""
        try:
            ai_id = ai_id or ("ai_1" if not self.turn else "ai_2")
            if ai_id not in self.ai_players:
                logger.error(f"Invalid AI ID: {ai_id}")
                return False

            ai = self.ai_players[ai_id]
            player_number = 1 if ai_id == "ai_1" else 2
            valid_moves = self._get_ai_valid_moves(ai)
            if not valid_moves:
                logger.info("No valid AI moves available")
                return False

            move = ai.get_move(self.board.board) or list(valid_moves.keys())[0]
            from_row, from_col, to_row, to_col = move

            if self.multi_jump_active and self.selected and (from_row, from_col) != self.selected:
                self.selected = (from_row, from_col)

            self.history.save_state(self)
            board_before = self.board.board.copy()
            if self.select(from_row, from_col) and self._move(to_row, to_col):
                board_after = self.board.board.copy()
                reward = self.reward_calculator.get_reward(player_number=player_number)
                ai.update(((from_row, from_col), (to_row, to_col)), reward, board_before, board_after)
                logger.info(f"AI move successful: {move}, reward: {reward}")
                return self.multi_jump_active and self.make_ai_move(ai_id) or True
            logger.error("AI move failed")
            return False
        except Exception as e:
            logger.error(f"Error in AI move (AI ID: {ai_id}): {e}")
            return False

    def _get_ai_valid_moves(self, ai):
        """دریافت حرکت‌های معتبر برای AI."""
        if self.multi_jump_active and self.selected:
            valid_moves = self.get_valid_moves(*self.selected)
            valid_moves = {(self.selected[0], self.selected[1], move[2], move[3]): skipped for move, skipped in valid_moves.items() if skipped}
        else:
            valid_moves = ai.get_valid_moves(self.board.board)
        return valid_moves

    def select(self, row, col):
        """انتخاب یک مهره."""
        try:
            piece = self.board.board[row, col]
            if piece != 0 and ((piece > 0 and not self.turn) or (piece < 0 and self.turn)):
                valid_moves = self.get_valid_moves(row, col)
                self.valid_moves = {(move[2], move[3]): skipped for move, skipped in valid_moves.items()}
                if valid_moves:
                    self.selected = (row, col)
                    logger.debug(f"Selected piece at ({row}, {col})")
                    return True
            logger.debug(f"Invalid selection at ({row}, {col})")
            return False
        except Exception as e:
            logger.error(f"Error selecting piece at ({row}, {col}): {e}")
            return False

    def _move(self, row, col):
        """انجام حرکت."""
        try:
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

                is_promotion = self._handle_promotion(piece, row, col, player_number)
                self._update_move_counters(player_number, is_capture, is_promotion)
                self._handle_multi_jump(row, col, skipped)

                self.board.player_1_left = sum(1 for row in self.board.board.flat if row > 0)
                self.board.player_2_left = sum(1 for row in self.board.board.flat if row < 0)
                logger.info(f"Move to ({row}, {col}) successful, player_1_left={self.board.player_1_left}, player_2_left={self.board.player_2_left}")
                return True
            logger.error(f"Invalid move to ({row}, {col})")
            return False
        except Exception as e:
            logger.error(f"Error moving to ({row}, {col}): {e}")
            return False

    def _handle_promotion(self, piece, row, col, player_number):
        """مدیریت ارتقاء مهره به شاه."""
        is_promotion = False
        if piece == 1 and row == 0 and not self.turn:
            self.board.board[row, col] = 2
            is_promotion = True
            logger.info(f"Promoted to king at ({row}, {col})")
        elif piece == -1 and row == self.board_size - 1 and self.turn:
            self.board.board[row, col] = -2
            is_promotion = True
            logger.info(f"Promoted to king at ({row}, {col})")
        return is_promotion

    def _update_move_counters(self, player_number, is_capture, is_promotion):
        """به‌روزرسانی شمارشگرهای حرکت."""
        self.total_moves[player_number] += 1
        self.consecutive_uniform_moves[player_number] = 0 if is_capture or is_promotion else self.consecutive_uniform_moves[player_number] + 1
        self.no_capture_or_king_moves = 0 if is_capture or is_promotion else self.no_capture_or_king_moves + 1

    def _handle_multi_jump(self, row, col, skipped):
        """مدیریت پرش‌های چندگانه."""
        self.valid_moves = {}
        self.multi_jump_active = False
        if skipped:
            self.selected = (row, col)
            next_moves = self.get_valid_moves(row, col)
            next_jumps = {(move[2], move[3]): skipped for move, skipped in next_moves.items() if skipped}
            if next_jumps:
                self.valid_moves = next_jumps
                self.multi_jump_active = True
                logger.info(f"Multi-jump active, next jumps: {next_jumps}")
            else:
                self.selected = None
                self.turn = not self.turn
                logger.info(f"No more jumps, turn changed to {self.turn}")
        else:
            self.selected = None
            self.turn = not self.turn
            logger.info(f"Simple move, turn changed to {self.turn}")

    def get_valid_moves(self, row, col):
        """محاسبه حرکت‌های مجاز."""
        try:
            piece = self.board.board[row, col]
            if piece == 0:
                logger.debug(f"No piece at ({row}, {col})")
                return {}

            player_number = 2 if piece < 0 else 1
            is_king = abs(piece) == 2
            all_jumps = self._get_all_jumps(player_number, is_king)
            if all_jumps:
                logger.debug(f"Jumps available: {all_jumps}")
                return all_jumps

            moves = {}
            directions = self.get_piece_directions(piece, is_king)
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < self.board_size and 0 <= new_col < self.board_size and self.board.board[new_row, new_col] == 0:
                    moves[(row, col, new_row, new_col)] = []
            logger.debug(f"Moves: {moves}")
            return moves
        except Exception as e:
            logger.error(f"Error getting valid moves for ({row}, {col}): {e}")
            return {}

    def _get_all_jumps(self, player_number, is_king):
        """دریافت تمام پرش‌های ممکن."""
        all_jumps = {}
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.board.board[r, c]
                if p != 0 and ((p < 0 and player_number == 2) or (p > 0 and player_number == 1)):
                    jumps = self._get_jumps_recursive(r, c, p, player_number, is_king, [], set())
                    for (end_row, end_col), skipped in jumps.items():
                        all_jumps[(r, c, end_row, end_col)] = skipped
        return all_jumps

    def _get_jumps_recursive(self, row, col, piece, player_number, is_king, skipped, visited):
        """محاسبه بازگشتی پرش‌ها."""
        try:
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
                                jump_row, jump_col, piece, player_number, is_king, new_skipped, visited.copy()
                            )
                            self.board.board[row, col] = original_piece
                            self.board.board[new_row, new_col] = original_target
                            self.board.board[jump_row, jump_col] = 0
                            jumps.update(next_jumps)
            return jumps
        except Exception as e:
            logger.error(f"Error in recursive jumps for ({row}, {col}): {e}")
            return {}

    @staticmethod
    def get_piece_directions(piece, is_king):
        """تعیین جهت‌های حرکت."""
        try:
            is_player_2 = piece < 0
            player_number = 2 if is_player_2 else 1
            if not is_king and player_number == 1:
                return [(-1, -1), (-1, 1)]
            elif not is_king and player_number == 2:
                return [(1, -1), (1, 1)]
            elif is_king:
                return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            return []
        except Exception as e:
            logger.error(f"Error getting piece directions: {e}")
            return []

    def update_game(self):
        """به‌روزرسانی وضعیت بازی."""
        if not self.paused:
            try:
                self._update_game()
            except Exception as e:
                logger.error(f"Error updating game: {e}")

    def _update_game(self):
        """به‌روزرسانی داخلی وضعیت بازی."""
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        """بررسی برنده یا تساوی."""
        try:
            if not self.game_started or self.game_over:
                return

            current_player_moves = self._has_valid_moves()
            player_number = 2 if self.turn else 1

            if self.consecutive_uniform_moves[1] >= self.max_uniform_moves and self.consecutive_uniform_moves[2] >= self.max_uniform_moves:
                self._end_game(None)
            elif self.total_moves[1] >= self.max_total_moves and self.total_moves[2] >= self.max_total_moves:
                self._end_game(None)
            elif self.no_capture_or_king_moves >= self.max_no_capture_moves:
                self._end_game(None)
            elif not current_player_moves:
                self._end_game(not self.turn)
            elif self.settings.use_timer:
                self._check_timer()
        except Exception as e:
            logger.error(f"Error checking winner: {e}")

    def _has_valid_moves(self):
        """بررسی وجود حرکت‌های معتبر."""
        player_number = 2 if self.turn else 1
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.board[row, col]
                if piece != 0 and ((piece > 0 and player_number == 1) or (piece < 0 and player_number == 2)):
                    if self.get_valid_moves(row, col):
                        return True
        return False

    def _check_timer(self):
        """بررسی تایمر بازی."""
        player_1_time = self.timer.get_current_time(False)
        player_2_time = self.timer.get_current_time(True)
        if player_1_time <= 0:
            self._end_game(True)
            self.time_up = True
        elif player_2_time <= 0:
            self._end_game(False)
            self.time_up = True

    def _end_game(self, winner):
        """پایان بازی و به‌روزرسانی امتیازات."""
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

    def handle_click(self, pos):
        """مدیریت کلیک کاربر."""
        if self.paused:
            return False
        try:
            x, y = pos
            self.click_log.append((x, y))
            if len(self.click_log) > 50:
                self.click_log.pop(0)

            if x < self.board_width and self.game_started and not self.game_over and self._is_human_turn():
                row = (y - self.menu_height - self.border_thickness) // self.square_size
                col = (x - self.border_thickness) // self.square_size
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    self.history.save_state(self)
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
        except Exception as e:
            logger.error(f"Error handling click at {pos}: {e}")
            return False

    def _is_human_turn(self):
        """بررسی نوبت بازیکن انسانی."""
        return self.settings.game_mode == "human_vs_human" or (self.settings.game_mode == "human_vs_ai" and not self.turn)

    def draw_valid_moves(self):
        """نمایش حرکت‌های معتبر."""
        try:
            drawn_moves = set()
            for move in self.valid_moves:
                row, col = move[2:4] if len(move) == 4 else move
                if (row, col) not in drawn_moves:
                    drawn_moves.add((row, col))
                    logger.debug(f"Drawing move at ({row}, {col})")
        except Exception as e:
            logger.error(f"Error drawing valid moves: {e}")

    def change_turn(self):
        """تغییر نوبت."""
        try:
            self.selected = None
            self.valid_moves = {}
            self.turn = not self.turn
            self.check_winner()
            logger.info(f"Turn changed to: {self.turn}")
        except Exception as e:
            logger.error(f"Error changing turn: {e}")

    def get_hint(self):
        """دریافت پیشنهاد حرکت."""
        try:
            if self.game_over or (not self.turn and not self.hint_enabled_p1) or (self.turn and not self.hint_enabled_p2):
                return None
            ai_id = "ai_1" if not self.turn else "ai_2"
            ai = self.ai_players.get(ai_id)
            if ai and self.settings.game_mode in ["human_vs_human", "human_vs_ai"]:
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        piece = self.board.board[row, col]
                        if piece != 0 and (piece < 0) == self.turn:
                            moves = self.get_valid_moves(row, col)
                            if moves:
                                move = getattr(ai, 'suggest_move', ai.get_move)(self.board.board)
                                if move:
                                    from_row, from_col, to_row, to_col = move
                                    return ((row, col), (to_row, to_col))
            return None
        except Exception as e:
            logger.error(f"Error getting hint: {e}")
            return None

    def save_game_state(self):
        """ذخیره وضعیت بازی."""
        self.history.save_state(self)