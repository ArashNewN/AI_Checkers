import pygame
import importlib
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from .checkers_core import get_piece_moves, make_move, log_to_json
from .board import Board
from .checkers_game import CheckersGame
from .settings import GameSettings
from .timer import TimerManager
from .rewards import RewardCalculator
from .config import load_stats, save_stats, load_ai_config, load_config, get_project_root
from .utils import CheckersError

# تنظیم لاگ
config = load_config()
logging.basicConfig(
    level=getattr(logging, config.get("logging_level", "DEBUG")),
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MoveHistory:
    # بدون تغییر
    def __init__(self):
        self.move_history = []
        self.redo_stack = []

    def save_state(self, game):
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
    def __init__(self, settings: Optional[GameSettings] = None, interface=None):
        self.board_size = config["board_size"]
        try:
            self.config = load_config()
            self.hovered_piece = None
            if isinstance(settings, GameSettings):
                self.settings = settings
            else:
                settings_dict = settings or {}
                for key, value in self.config.items():
                    if key not in settings_dict:
                        settings_dict[key] = value
                assets_dir = settings_dict.get("assets_dir", "assets")
                project_root = get_project_root()
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
                    "player_1_king_image", "player_2_king_image", "assets_dir"
                }
                filtered_settings = {k: v for k, v in settings_dict.items() if k in valid_settings_keys}
                logger.debug(f"Settings passed to GameSettings: {filtered_settings}")
                self.settings = GameSettings(**filtered_settings)
            if not hasattr(self.settings, "game_mode") or not self.settings.game_mode:
                setattr(self.settings, "game_mode", "ai_vs_ai")
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
            self.ai_players: Dict[str, 'BaseAI'] = {}  # تایپ برای ai_players
            self._init_ai_players()
            logger.info("Game initialized")
        except Exception as e:
            logger.error(f"Error initializing game: {e}")
            raise

    def _init_fonts(self):
        font_name = 'Vazir' if 'Vazir' in pygame.font.get_fonts() else 'Arial'
        self.font = pygame.font.SysFont(font_name, 24)
        self.small_font = pygame.font.SysFont(font_name, 18)

    def _init_stats(self):
        stats = load_stats()
        self.player_1_wins = stats.get("player_1_wins", 0)
        self.player_2_wins = stats.get("player_2_wins", 0)

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
            log_to_json(f"Error initializing AI players: {str(e)}", level="ERROR")
            self.ai_players = {}  # Clear AI players on error

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
                logger.error("No valid AI classes loaded")
                log_to_json("No valid AI classes loaded", level="ERROR")
                return
            players = []
            game_mode = getattr(self.settings, "game_mode", "human_vs_human")
            player_1_ai_type = getattr(self.settings, "player_1_ai_type", "none")
            player_2_ai_type = getattr(self.settings, "player_2_ai_type", "none")
            logger.debug(f"Game mode: {game_mode}, Player 1 AI type: {player_1_ai_type}, Player 2 AI type: {player_2_ai_type}")
            if game_mode == "ai_vs_ai":
                players = ["player_1", "player_2"]
            elif game_mode == "human_vs_ai":
                players = ["player_2"]
            elif player_1_ai_type != "none":
                players = ["player_1"]
            elif player_2_ai_type != "none":
                players = ["player_2"]
            logger.debug(f"Players to initialize: {players}")
            self.ai_players = {}  # Clear existing AI players
            for player in players:
                ai_id = "ai_1" if player == "player_1" else "ai_2"
                ai_settings = config_dict["ai_configs"].get(player, {})
                ai_type = ai_settings.get("ai_type", "advanced_ai")
                model_name = ai_settings.get("ai_code", ai_type)
                logger.debug(f"Initializing AI for {player}: ai_id={ai_id}, ai_type={ai_type}, model_name={model_name}")
                if ai_type and ai_type != "none":
                    ai_class = ai_type_to_class.get(ai_type)
                    if ai_class:
                        try:
                            self.ai_players[ai_id] = ai_class(
                                game=self.checkers_game,
                                model_name=model_name,
                                ai_id=ai_id
                            )
                            logger.info(f"Loaded AI: {ai_type} for {player} (ID: {ai_id})")
                        except Exception as e:
                            logger.error(f"Error loading AI {ai_type} for {ai_id}: {e}")
                            log_to_json(f"Error loading AI {ai_type} for {ai_id}: {str(e)}", level="ERROR", extra_data={"ai_id": ai_id})
                    else:
                        logger.error(f"No AI class found for type: {ai_type}")
                        log_to_json(f"No AI class found for type: {ai_type}", level="ERROR", extra_data={"ai_type": ai_type})
        except Exception as e:
            logger.error(f"Error updating AI players: {e}")
            log_to_json(f"Error updating AI players: {str(e)}", level="ERROR")
            self.ai_players = {}  # Clear AI players on error

    def make_ai_move(self, ai_id: str) -> bool:
        try:
            move: Optional[Tuple[int, int, int, int]] = None
            ai = self.ai_players.get(ai_id)
            if not ai:
                log_to_json(
                    f"No AI found for {ai_id}. Available AI players: {list(self.ai_players.keys())}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "available_ai_players": list(self.ai_players.keys())}
                )
                return False
            player_number = 1 if ai_id == "ai_1" else 2
            player = 1 if player_number == 1 else -1
            log_to_json(
                f"Attempting AI move for {ai_id}, multi_jump_active={self.multi_jump_active}, selected={self.selected}",
                level="DEBUG",
                extra_data={"ai_id": ai_id, "board": self.board.board.tolist()}
            )
            valid_moves = self._get_ai_valid_moves(ai)
            log_to_json(
                f"Valid moves for {ai_id}: {list(valid_moves.keys())}",
                level="DEBUG",
                extra_data={"ai_id": ai_id, "valid_moves": [list(move) for move in valid_moves.keys()]}
            )
            if not valid_moves:
                log_to_json(
                    f"No valid moves available for {ai_id}",
                    level="WARNING",
                    extra_data={"ai_id": ai_id, "board": self.board.board.tolist()}
                )
                self.turn = not self.turn
                return False
            valid_moves_formatted = list(valid_moves.keys())
            move = ai.get_move(self.board.board)
            log_to_json(
                f"Move returned by {ai_id}: {move}",
                level="DEBUG",
                extra_data={"ai_id": ai_id, "move": list(move) if move else None}
            )
            if not move or not isinstance(move, tuple) or len(move) != 4:
                log_to_json(
                    f"Invalid move format returned by {ai_id}: {move}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "move": list(move) if move else None,
                                "valid_moves": [list(m) for m in valid_moves_formatted]}
                )
                move = valid_moves_formatted[0]
            if move not in valid_moves_formatted:
                log_to_json(
                    f"Move {move} not in valid moves for {ai_id}",
                    level="ERROR",
                    extra_data={"ai_id": ai_id, "move": list(move),
                                "valid_moves": [list(m) for m in valid_moves_formatted]}
                )
                move = valid_moves_formatted[0]
            from_row, from_col, to_row, to_col = move
            log_to_json(
                f"Validating move {move} for player {player}",
                level="DEBUG",
                extra_data={"ai_id": ai_id, "move": list(move), "board": self.board.board.tolist()}
            )
            board_before = self.board.copy()
            new_board, is_promotion, has_more_jumps = make_move(self.board, move, player_number)
            if new_board is None:
                log_to_json(
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
            log_to_json(
                f"Updating AI {ai_id} with move {move}, reward {reward}",
                level="DEBUG",
                extra_data={"ai_id": ai_id, "move": list(move)}
            )
            ai.update(move, reward, board_before.board, board_after.board)
            is_jump = abs(to_row - from_row) == 2
            self._handle_multi_jump(to_row, to_col, is_jump, is_promotion, player_number)
            log_to_json(
                f"AI move executed: {move}",
                level="INFO",
                extra_data={"ai_id": ai_id, "move": list(move)}
            )
            if self.interface:
                piece_value = board_before.board[from_row, from_col]
                self.interface.animate_move(piece_value, from_row, from_col, to_row, to_col, is_kinged=is_promotion)
            return True
        except Exception as e:
            log_to_json(
                f"Error in AI move for {ai_id}: {str(e)}",
                level="ERROR",
                extra_data={
                    "ai_id": ai_id,
                    "move": list(move) if move is not None else None,
                    "valid_moves": [list(m) for m in
                                    valid_moves_formatted] if 'valid_moves_formatted' in locals() else None,
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
            log_to_json(
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
                move = getattr(ai, 'suggest_move', ai.get_move)(self.board.board)
                if move and isinstance(move, tuple) and len(move) == 4:
                    return move  # Return (from_row, from_col, to_row, to_col)
            return None
        except Exception as e:
            logger.error(f"Error getting hint: {e}")
            log_to_json(f"Error getting hint: {str(e)}", level="ERROR")
            return None

    def _handle_multi_jump(self, row, col, is_jump, is_promotion, player_number):
        try:
            self.valid_moves = {}
            self.multi_jump_active = False
            player = 1 if player_number == 1 else -1
            if not is_jump or is_promotion:
                self.selected = None
                self.turn = not self.turn
                log_to_json(
                    f"{'Simple move' if not is_jump else 'Jump move'}, turn switched to {'Player 2' if self.turn else 'Player 1'}",
                    level="INFO",
                    extra_data={"player_number": player_number, "is_jump": is_jump, "is_promotion": is_promotion}
                )
            else:
                additional_jumps = get_piece_moves(self.board, row, col, player)
                jump_moves = {(row, col, to_row, to_col): skipped for (to_row, to_col), skipped in
                              additional_jumps.items() if
                              abs(to_row - row) == 2}
                if jump_moves:
                    self.selected = (row, col)
                    self.valid_moves = jump_moves
                    self.multi_jump_active = True
                    log_to_json(
                        f"Multi-jump active for player {player_number} at ({row}, {col}) with moves: {list(jump_moves.keys())}",
                        level="INFO",
                        extra_data={"player_number": player_number,
                                    "valid_moves": [list(move) for move in jump_moves.keys()]}
                    )
                else:
                    self.selected = None
                    self.turn = not self.turn
                    log_to_json(
                        f"No additional jumps for player {player_number}, turn switched",
                        level="INFO",
                        extra_data={"player_number": player_number}
                    )
        except Exception as e:
            logger.error(f"Error in _handle_multi_jump: {e}")
            log_to_json(
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
            log_to_json(
                f"Turn changed to {'Player 2' if self.turn else 'Player 1'}",
                level="INFO",
                extra_data={"turn": self.turn}
            )
        except Exception as e:
            logger.error(f"Error changing turn: {e}")
            log_to_json(f"Error changing turn: {str(e)}", level="ERROR")

    def get_valid_moves(self, board: np.ndarray) -> Dict[Tuple[int, int, int, int], List[Tuple[int, int]]]:
        try:
            temp_game = self.checkers_game.copy()
            temp_game.set_state(board)
            temp_game.current_player = 1 if not self.turn else -1
            legal_moves = temp_game.get_legal_moves()
            valid_moves = {(move[0], move[1], move[2], move[3]): [] for move in legal_moves if len(move) == 4}
            log_to_json(
                f"Valid moves retrieved: {valid_moves}",
                level="DEBUG",
                extra_data={"player": temp_game.current_player, "move_count": len(valid_moves)}
            )
            return valid_moves
        except Exception as e:
            log_to_json(
                f"Error in get_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"board": board.tolist()}
            )
            raise CheckersError(f"Failed to get valid moves: {str(e)}")

    @staticmethod
    def _load_ai_classes(config_dict):
        ai_type_to_class = {}
        for ai_type, ai_info in config_dict.get("ai_types", {}).items():
            try:
                module_name = ai_info.get("module")
                class_name = ai_info.get("class")
                if not module_name or not class_name:
                    logger.error(f"Missing module or class for AI type: {ai_type}")
                    log_to_json(f"Missing module or class for AI type: {ai_type}", level="ERROR")
                    continue
                module = importlib.import_module(module_name)
                ai_class = getattr(module, class_name)
                if not isinstance(ai_class, type):
                    logger.error(f"{class_name} is not a class")
                    log_to_json(f"{class_name} is not a class", level="ERROR")
                    continue
                ai_type_to_class[ai_type] = ai_class
                logger.debug(f"Mapped {ai_type} to {class_name}")
            except Exception as e:
                logger.error(f"Error importing AI class {class_name if 'class_name' in locals() else 'unknown'}: {e}")
                log_to_json(
                    f"Error importing AI class {class_name if 'class_name' in locals() else 'unknown'}: {str(e)}",
                    level="ERROR")
        return ai_type_to_class

    def select(self, row, col):
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
                    self.valid_moves = {(row, col, to_row, to_col): skipped for (to_row, to_col), skipped in
                                        valid_moves.items()}
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

    def _move(self, move: Tuple[int, int, int, int]):
        try:
            if self.paused or not self.game_started:
                log_to_json(
                    "Cannot move while paused or game not started",
                    level="DEBUG",
                    extra_data={"paused": self.paused, "game_started": self.game_started}
                )
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
            log_to_json(
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
                log_to_json(
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
            log_to_json(
                f"Game ended, winner: {self.winner}",
                level="INFO",
                extra_data={"player_1_wins": self.player_1_wins, "player_2_wins": self.player_2_wins}
            )

    def handle_click(self, pos):
        if self.paused:
            log_to_json("Cannot handle click while paused", level="DEBUG", extra_data={"paused": self.paused})
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
                    return False
                if self.selected:
                    move = (self.selected[0], self.selected[1], row, col)
                    if (move[2], move[3]) in [(m[2], m[3]) for m in self.valid_moves]:
                        self.history.save_state(self)
                        return self._move(move)
                result = self.select(row, col)
                if not result and self.multi_jump_active and self.selected:
                    self.valid_moves = {(self.selected[0], self.selected[1], to_row, to_col): skipped for
                                        (to_row, to_col), skipped in
                                        get_piece_moves(self.board, *self.selected, 1 if not self.turn else -1).items()}
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
                    log_to_json(
                        f"Invalid move format in valid_moves: {move}",
                        level="ERROR",
                        extra_data={"move": list(move) if isinstance(move, tuple) else move}
                    )
                    continue
                _, _, row, col = move
                if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                    log_to_json(
                        f"Invalid move position: ({row}, {col})",
                        level="ERROR",
                        extra_data={"move": [row, col]}
                    )
                    continue
                x = border_thickness + col * square_size + square_size // 2
                y = menu_height + border_thickness + row * square_size + square_size // 2
                surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                pygame.draw.circle(surface, valid_move_color, (square_size // 2, square_size // 2), circle_radius)
                self.screen.blit(surface, (x - square_size // 2, y - square_size // 2))
                log_to_json(f"Drawing valid move at ({row}, {col})", level="DEBUG", extra_data={"move": [row, col]})
            if hovered_moves:
                for move, _ in hovered_moves.items():
                    if not (isinstance(move, tuple) and len(move) == 4 and all(isinstance(x, int) for x in move)):
                        log_to_json(
                            f"Invalid move format in hovered_moves: {move}",
                            level="ERROR",
                            extra_data={"move": list(move) if isinstance(move, tuple) else move}
                        )
                        continue
                    _, _, row, col = move
                    if not (0 <= row < self.board.board_size and 0 <= col < self.board.board_size):
                        log_to_json(
                            f"Invalid hovered move position: ({row}, {col})",
                            level="ERROR",
                            extra_data={"move": [row, col]}
                        )
                        continue
                    x = border_thickness + col * square_size + square_size // 2
                    y = menu_height + border_thickness + row * square_size + square_size // 2
                    surface = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
                    pygame.draw.circle(surface, valid_move_color, (square_size // 2, square_size // 2), circle_radius)
                    self.screen.blit(surface, (x - square_size // 2, y - square_size // 2))
                    log_to_json(f"Drawing hovered move at ({row}, {col})", level="DEBUG",
                                extra_data={"move": [row, col]})
        except Exception as e:
            log_to_json(
                f"Error in draw_valid_moves: {str(e)}",
                level="ERROR",
                extra_data={"valid_moves": [list(k) for k in self.valid_moves.keys()] if self.valid_moves else None,
                            "hovered_moves": [list(k) for k in hovered_moves.keys()] if hovered_moves else None}
            )

    def save_game_state(self):
        self.history.save_state(self)