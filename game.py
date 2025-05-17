# game.py
import pygame
import importlib
import logging
from .checkers_core import get_piece_moves, make_move, log_to_json
from .board import Board
from .settings import GameSettings
from .timer import TimerManager
from pathlib import Path
from .rewards import RewardCalculator
from .config import load_stats, save_stats, load_ai_config, load_config
from .utils import CheckersError

# Load configuration
config = load_config()
logging.basicConfig(
    level=getattr(logging, config.get("logging_level", "ERROR")),
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MoveHistory:
    """Manages the history of game states for undo/redo functionality.

    Attributes:
        move_history (list): List of saved game states.
        redo_stack (list): Stack of states for redo.
    """
    def __init__(self):
        self.move_history = []
        self.redo_stack = []

    def save_state(self, game):
        """Saves the current game state."""
        try:
            state = {
                'board': game.board.board.copy(),
                'turn': game.turn,
                'player_1_left': game.board.player_1_left,
                'player_2_left': game.board.player_2_left,
                'multi_jump_active': game.multi_jump_active,
                'selected': game.selected,
                'valid_moves': game.valid_moves.copy(),
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
        """Restores the previous game state."""
        try:
            if not self.move_history:
                logger.info("No moves to undo")
                return
            current_state = {
                'board': game.board.board.copy(),
                'turn': game.turn,
                'player_1_left': game.board.player_1_left,
                'player_2_left': game.board.player_2_left,
                'multi_jump_active': game.multi_jump_active,
                'selected': game.selected,
                'valid_moves': game.valid_moves.copy(),
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
    """Main class for the Checkers game.

    Attributes:
        board (Board): Current board state.
        settings: Game settings.
        interface: User interface.
        turn (bool): False for player 1, True for player 2.
        game_over (bool): Whether the game is over.
    """

    def __init__(self, settings=None, interface=None):
        try:
            self.config = load_config()
            if isinstance(settings, GameSettings):
                self.settings = settings
            else:
                settings_dict = settings or {}
                for key, value in self.config.items():
                    if key not in settings_dict:
                        settings_dict[key] = value
                assets_dir = settings_dict.get("assets_dir", "assets")
                # اعمال تنظیمات گرافیکی پیش‌فرض اگر خالی باشند
                settings_dict["player_1_piece_image"] = settings_dict.get("player_1_piece_image") or str(
                    Path(assets_dir) / "white_piece.png")
                settings_dict["player_2_piece_image"] = settings_dict.get("player_2_piece_image") or str(
                    Path(assets_dir) / "black_piece.png")
                settings_dict["player_1_king_image"] = settings_dict.get("player_1_king_image") or str(
                    Path(assets_dir) / "white_king.png")
                settings_dict["player_2_king_image"] = settings_dict.get("player_2_king_image") or str(
                    Path(assets_dir) / "black_king.png")
                self.settings = GameSettings(**settings_dict)
            # اطمینان از وجود game_mode و player_1_ai_type
            if not hasattr(self.settings, "game_mode") or not self.settings.game_mode:
                setattr(self.settings, "game_mode", "ai_vs_ai")  # یا "human_vs_ai" بسته به نیاز
            if not hasattr(self.settings, "player_1_ai_type") or not self.settings.player_1_ai_type:
                setattr(self.settings, "player_1_ai_type", "advanced_ai")
            self.interface = interface
            self.screen = pygame.display.set_mode((
                getattr(self.settings, "window_width", 940),
                getattr(self.settings, "window_height", 720)
            ))
            pygame.display.set_caption('Checkers: Player 1 vs Player 2')
            self._init_fonts()
            self._init_stats()
            self.board = Board(self.settings)
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
            self._init_ai_players()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")
            raise

    def _init_fonts(self):
        """Initializes fonts for the game."""
        font_name = 'Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial'
        self.font = pygame.font.SysFont(font_name, 24)
        self.small_font = pygame.font.SysFont(font_name, 18)

    def _init_stats(self):
        """Loads game statistics."""
        stats = load_stats()
        self.player_1_wins = stats.get("player_1_wins", 0)
        self.player_2_wins = stats.get("player_2_wins", 0)

    def _init_ai_players(self):
        """Initializes AI players."""
        ai_config = load_ai_config()
        self.ai_players = {}
        game_mode = getattr(self.settings, "game_mode", "human_vs_human")
        player_1_ai_type = getattr(self.settings, "player_1_ai_type", "none")
        player_2_ai_type = getattr(self.settings, "player_2_ai_type", "none")
        if game_mode in ["human_vs_ai", "ai_vs_ai"] or player_1_ai_type != "none" or player_2_ai_type != "none":
            self.reward_calculator = RewardCalculator(self)
        else:
            self.reward_calculator = None
        self.update_ai_players()

    def update_ai_players(self):
        """Updates AI players based on configuration."""
        try:
            config_dict = load_ai_config()
            # افزودن تنظیمات گرافیکی به config_dict
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
                logger.error("No valid AI classes loaded")
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
            for player in players:
                ai_id = "ai_1" if player == "player_1" else "ai_2"
                ai_settings = config_dict["ai_configs"].get(player, {})
                ai_type = ai_settings.get("ai_type")
                model_name = ai_settings.get("ai_code", "default")
                logger.debug(f"Initializing AI for {player}: ai_id={ai_id}, ai_type={ai_type}, model_name={model_name}")
                if ai_type and ai_type != "none":
                    ai_class = ai_type_to_class.get(ai_type)
                    if ai_class:
                        self.ai_players[ai_id] = ai_class(
                            game=self,
                            model_name=model_name,
                            ai_id=ai_id,
                            config=config_dict
                        )
                        logger.info(f"Loaded AI: {ai_type} for {player} (ID: {ai_id})")
                    else:
                        logger.error(f"No AI class found for type: {ai_type}")
        except Exception as e:
            logger.error(f"Error updating AI players: {str(e)}")
            raise  # برای دیباگ، استثنا را پرتاب می‌کنیم

    def _load_ai_classes(self, config_dict):
        """Loads AI classes from configuration."""
        ai_type_to_class = {}
        for ai_info in config_dict.get("available_ais", []):
            try:
                module_name = ai_info.get("module")
                class_name = ai_info.get("class", "UnknownClass")
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
        """Initializes the game state."""
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.timer.start_game()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")

    def start_game(self):
        """Starts a new game."""
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
        """Resets the game board."""
        try:
            self._reset_game_state()
            self.board = Board(self.settings)
            self.timer.start_game()
            self.update_ai_players()
            logger.info("Board reset")
        except Exception as e:
            logger.error(f"Error resetting board: {e}")

    def _reset_game_state(self):
        """Resets the game state."""
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

    def make_ai_move(self, ai_id=None):
        try:
            ai = self.ai_players.get(ai_id)
            if not ai:
                log_to_json(
                    f"No AI found for {ai_id}. Available AI players: {list(self.ai_players.keys())}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id}
                )
                return False
            player_number = 1 if ai_id == "ai_1" else 2
            player = 1 if player_number == 1 else -1
            move = ai.get_move(self.board.board)
            if not move:
                log_to_json(
                    f"No valid move returned by {ai_id}",
                    level="WARNING",
                    extra_data={"ai_id": ai_id}
                )
                return False
            from_row, from_col, to_row, to_col = move
            board_before = self.board.copy()
            new_board, is_promotion, has_more_jumps = make_move(self.board, move, player)
            if new_board is None:
                log_to_json(
                    f"Invalid AI move: {move}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "move": move}
                )
                return False
            self.board = new_board
            board_after = self.board.copy()
            reward = self.reward_calculator.get_reward(player_number=player_number)
            ai.update(((from_row, from_col), (to_row, to_col)), reward, board_before.board, board_after.board)
            is_jump = abs(to_row - from_row) == 2
            self._handle_multi_jump(to_row, to_col, is_jump, is_promotion, player_number)
            log_to_json(
                f"AI move executed: {move}",
                level="INFO",
                extra_data={"ai_id": ai_id, "move": move}
            )
            return True
        except Exception as e:
            log_to_json(
                f"Error in AI move for {ai_id}: {str(e)}",
                level="ERROR",
                extra_data={"ai_id": ai_id, "move": move if 'move' in locals() else None}
            )
            return False

    def _get_ai_valid_moves(self, ai):
        """Gets valid moves for the AI."""
        player = 1 if not self.turn else -1
        if self.multi_jump_active and self.selected:
            return get_piece_moves(self.board, *self.selected, player)
        return ai.get_valid_moves(self.board.board)

    def select(self, row, col):
        """Selects a piece for movement."""
        try:
            if self.paused or not self.game_started:
                log_to_json(
                    "Cannot select piece while paused or game not started",
                    level="DEBUG",
                    extra_data={"paused": self.paused, "game_started": self.game_started}
                )
                return False
            piece = self.board.board[row, col]
            player = 1 if not self.turn else -1
            if piece != 0 and piece * player > 0:
                valid_moves = get_piece_moves(self.board, row, col, player)
                if valid_moves:
                    self.selected = (row, col)
                    self.valid_moves = {(to_row, to_col): skipped for (to_row, to_col), skipped in valid_moves.items()}
                    logger.debug(f"Selected piece at ({row}, {col}) with valid moves: {self.valid_moves}")
                    return True
                logger.debug(f"No valid moves for piece at ({row}, {col})")
            else:
                logger.debug(f"Invalid selection at ({row}, {col}): piece={piece}, player={player}")
            return False
        except Exception as e:
            logger.error(f"Error selecting piece at ({row}, {col}): {e}")
            log_to_json(f"Error selecting piece at ({row}, {col}): {str(e)}", level="ERROR",
                        extra_data={"row": row, "col": col})
            return False

    def _move(self, row, col):
        """Performs a move."""
        try:
            if self.paused or not self.game_started:
                log_to_json(
                    "Cannot move while paused or game not started",
                    level="DEBUG",
                    extra_data={"paused": self.paused, "game_started": self.game_started}
                )
                return False
            if self.selected and (row, col) in self.valid_moves:
                start_row, start_col = self.selected
                player_number = 1 if not self.turn else 2
                player = 1 if player_number == 1 else -1
                move = (start_row, start_col, row, col)
                self.history.save_state(self)
                new_board, is_promotion, has_more_jumps = make_move(self.board, move, player_number)
                if new_board is None:
                    logger.error(f"Invalid move to ({row}, {col})")
                    return False
                self.board = new_board
                is_jump = abs(row - start_row) == 2
                self._update_move_counters(player_number, is_jump, is_promotion)
                self._handle_multi_jump(row, col, is_jump, is_promotion, player_number)
                logger.info(
                    f"Move to ({row}, {col}) successful, player_1_left={self.board.player_1_left}, player_2_left={self.board.player_2_left}"
                )
                return True
            logger.error(f"Invalid move to ({row}, {col})")
            return False
        except CheckersError as e:
            logger.error(f"CheckersError moving to ({row}, {col}): {e}")
            return False
        except Exception as e:
            logger.error(f"Error moving to ({row}, {col}): {e}")
            return False

    def _update_move_counters(self, player_number, is_capture, is_promotion):
        """Updates move counters."""
        self.total_moves[player_number] += 1
        self.consecutive_uniform_moves[player_number] = 0 if is_capture or is_promotion else \
            self.consecutive_uniform_moves[player_number] + 1
        self.no_capture_or_king_moves = 0 if is_capture or is_promotion else self.no_capture_or_king_moves + 1

    def _handle_multi_jump(self, row, col, is_jump, is_promotion, player_number):
        """Handles multi-jump logic."""
        try:
            self.valid_moves = {}
            self.multi_jump_active = False
            player = 1 if player_number == 1 else -1
            if not is_jump:
                self.selected = None
                self.turn = not self.turn
                log_to_json(
                    f"Simple move completed, turn changed to {self.turn}",
                    level="INFO",
                    extra_data={"position": (row, col)}
                )
                return
            self.selected = (row, col)
            if is_promotion:
                self.selected = None
                self.turn = not self.turn
                log_to_json(
                    f"Piece promoted to king at ({row}, {col}), turn changed to {self.turn}",
                    level="INFO",
                    extra_data={"player_number": player_number}
                )
                return
            next_jumps = get_piece_moves(self.board, row, col, player)
            if next_jumps:
                self.valid_moves = {(m[2], m[3]): [] for m in next_jumps}
                self.multi_jump_active = True
                log_to_json(
                    f"Multi-jump active, next jumps: {next_jumps}",
                    level="INFO",
                    extra_data={"position": (row, col)}
                )
            else:
                self.selected = None
                self.turn = not self.turn
                log_to_json(
                    f"No more jumps available, turn changed to {self.turn}",
                    level="INFO",
                    extra_data={"position": (row, col)}
                )
        except Exception as e:
            log_to_json(
                f"Error in _handle_multi_jump: {str(e)}",
                level="ERROR",
                extra_data={"row": row, "col": col, "board": self.board.board.tolist()}
            )
            self.selected = None
            self.turn = not self.turn

    def get_valid_moves(self, row, col):
        """Calculates valid moves for a piece."""
        try:
            if self.paused or not self.game_started:
                log_to_json(
                    "Cannot get valid moves while paused or game not started",
                    level="DEBUG",
                    extra_data={"paused": self.paused, "game_started": self.game_started}
                )
                return {}
            player = 1 if not self.turn else -1
            moves = get_piece_moves(self.board, row, col, player)
            return {(m[0], m[1], m[2], m[3]): [] for m in moves}
        except Exception as e:
            log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"row": row, "col": col, "board": self.board.board.tolist()}
            )
            return {}

    def update_game(self):
        """Updates the game state."""
        if not self.paused:
            try:
                self._update_game()
            except Exception as e:
                logger.error(f"Error updating game: {e}")
                log_to_json(
                    f"Error updating game: {str(e)}",
                    level="ERROR",
                    extra_data={"game_started": self.game_started, "game_over": self.game_over}
                )

    def _update_game(self):
        """Internal game state update."""
        if self.game_started and not self.game_over:
            self.timer.update(self.turn, self.game_started, self.game_over)
            self.check_winner()

    def check_winner(self):
        """Checks if the game has a winner."""
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
                            return None  # بازی ادامه دارد
            self.game_over = True
            self.winner = not self.turn  # برنده بازیکن دیگر است
            log_to_json(
                f"Game over, winner: {'black' if self.winner else 'white'}",
                level="INFO"
            )
            return self.winner
        except Exception as e:
            log_to_json(
                f"Error checking winner: {str(e)}",
                level="ERROR"
            )
            return None

    def _has_valid_moves(self):
        """Checks if the current player has valid moves."""
        player = 1 if not self.turn else -1
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if (row + col) % 2 == 1 and self.board.board[row, col] * player > 0:
                    if get_piece_moves(self.board, row, col, player):
                        return True
        return False

    def _check_timer(self):
        """Checks the game timer."""
        player_1_time = self.timer.get_current_time(False)
        player_2_time = self.timer.get_current_time(True)
        if player_1_time <= 0:
            self._end_game(True)
            self.time_up = True
        elif player_2_time <= 0:
            self._end_game(False)
            self.time_up = True

    def _end_game(self, winner):
        """Ends the game and updates scores."""
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
            logger.autobus.info(f"Game ended, winner: {self.winner}")
            log_to_json(
                f"Game ended, winner: {self.winner}",
                level="INFO",
                extra_data={"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins}
            )

    def handle_click(self, pos):
        """Handles user clicks."""
        if self.paused:
            log_to_json("Cannot handle click while paused", level="DEBUG", extra_data={"paused": self.paused})
            return False
        try:
            x, y = pos
            self.click_log.append((x, y))
            if len(self.click_log) > 50:
                self.click_log.pop(0)
            if x < self.config.get("board_width") and self.game_started and not self.game_over and self._is_human_turn():
                row = (y - self.config.get("menu_height") - self.config.get("border_thickness")) // self.config.get("square_size")
                col = (x - self.config.get("border_thickness")) // self.config.get("square_size")
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    return False
                if self.selected and (row, col) in self.valid_moves:
                    self.history.save_state(self)
                    return self._move(row, col)
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    self.valid_moves = {(m[2], m[3]): [] for m in get_piece_moves(self.board, *self.selected, 1 if not self.turn else -1)}
                return result
            if self.game_over:
                self.selected = None
                self.valid_moves = {}
                self.multi_jump_active = False
            return False
        except Exception as e:
            logger.error(f"Error handling click at {pos}: {e}")
            log_to_json(f"Error handling click at {pos}: {str(e)}", level="ERROR", extra_data={"pos": pos})
            return False

    def _is_human_turn(self):
        """Checks if it's a human player's turn."""
        return self.settings.game_mode == "human_vs_human" or (
            self.settings.game_mode == "human_vs_ai" and not self.turn)

    def draw_valid_moves(self):
        """Draws valid moves on the board."""
        try:
            if not self.game_started or self.paused:
                return
            for move in self.valid_moves:
                row, col = move
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    log_to_json(
                        f"Invalid move position: ({row}, {col})",
                        level="ERROR",
                        extra_data={"move": move}
                    )
                    continue
                log_to_json(f"Drawing move at ({row}, {col})", level="DEBUG", extra_data={"move": move})
        except Exception as e:
            log_to_json(
                f"Error in draw_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"valid_moves": list(self.valid_moves.keys())}
            )

    def change_turn(self):
        """Changes the current turn."""
        try:
            self.selected = None
            self.valid_moves = {}
            self.turn = not self.turn
            self.check_winner()
            logger.info(f"Turn changed to: {self.turn}")
            log_to_json(f"Turn changed to: {self.turn}", level="INFO")
        except Exception as e:
            logger.error(f"Error changing turn: {e}")
            log_to_json(f"Error changing turn: {str(e)}", level="ERROR")

    def get_hint(self):
        """Returns a move suggestion."""
        try:
            if self.game_over or (not self.turn and not self.config.get("hint_enabled_p1")) or (
                    self.turn and not self.config.get("hint_enabled_p2")):
                return None
            ai_id = "ai_1" if not self.turn else "ai_2"
            ai = self.ai_players.get(ai_id)
            if ai and self.settings.game_mode in ["human_vs_human", "human_vs_ai"]:
                move = getattr(ai, 'suggest_move', ai.get_move)(self.board.board)
                if move and len(move) == 4:
                    from_row, from_col, to_row, to_col = move
                    return (from_row, from_col), (to_row, to_col)
            return None
        except Exception as e:
            logger.error(f"Error getting hint: {e}")
            log_to_json(f"Error getting hint: {str(e)}", level="ERROR")
            return None

    def save_game_state(self):
        """Saves the current game state."""
        self.history.save_state(self)