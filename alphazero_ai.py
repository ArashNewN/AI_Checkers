import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from .checkers_game import CheckersGame
from .mcts import MCTS
from .alphazero_net import AlphaZeroNet
from .config import load_config

# تنظیم لاگ‌گیری
log_dir = os.path.dirname(os.path.dirname(__file__))
log_file = os.path.join(log_dir, 'alphazero_ai.log')
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)
logger = logging.getLogger(__name__)


class AlphaZeroAI:
    def __init__(self, game=None, neural_net=None, color=None, model_name=None, az_id=None):
        config = load_config()
        # بررسی وجود training_params
        training_params_key = f"{az_id}_training_params"
        if training_params_key not in config:
            logger.error(f"Missing {training_params_key} in config")
            raise KeyError(f"Configuration key {training_params_key} not found in config")
        training_params = config[training_params_key]
        self.game = game if game is not None else CheckersGame()
        self.color = color
        self.model_name = model_name
        self.az_id = az_id
        # انتخاب ability بر اساس az_id
        ability_key = f"{az_id}_ability"
        self.ability = config.get(ability_key, 5)
        self.epsilon = training_params["epsilon_start"]
        self.epsilon_end = training_params["epsilon_end"]
        self.epsilon_decay = training_params["epsilon_decay"]
        self.neural_net = neural_net if neural_net is not None else AlphaZeroNet().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.mcts = MCTS(self, CheckersGame(), self.neural_net)
        self.memory = []
        self.reward_calculator = RewardCalculator()
        logger.debug(f"Initialized AlphaZeroAI with az_id={az_id}, ability={self.ability}")

    def set_game(self, game):
        self.game = game

    def get_state(self, board):
        config = load_config()
        board_size = config["network_params"]["board_size"]
        state = np.zeros((4, board_size, board_size), dtype=np.float32)
        for row in range(board_size):
            for col in range(board_size):
                piece = board.board[row][col]
                if piece and hasattr(piece, 'is_player_2'):
                    if piece.is_player_2:
                        state[2, row, col] = 1.0 if not piece.king else 0.0
                        state[3, row, col] = 1.0 if piece.king else 0.0
                    else:
                        state[0, row, col] = 1.0 if not piece.king else 0.0
                        state[1, row, col] = 1.0 if piece.king else 0.0
        logger.debug(f"State generated: shape={state.shape}")
        return state

    def convert_state_to_network_format(self, state):
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        logger.debug(f"Experience saved: action={action}, reward={reward}")

    def replay(self):
        config = load_config()
        training_params = config[f"{self.az_id}_training_params"]
        batch_size = training_params["replay_batch_size"]
        if len(self.memory) < batch_size:
            logger.debug("Not enough memory for replay")
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states).to(self.neural_net.device)
        actions = torch.LongTensor(actions).to(self.neural_net.device)
        rewards = torch.FloatTensor(rewards).to(self.neural_net.device)
        next_states = torch.FloatTensor(next_states).to(self.neural_net.device)
        dones = torch.FloatTensor(dones).to(self.neural_net.device)

        policy, value = self.neural_net(states)
        target_policy, target_value = self._compute_targets(next_states, rewards, dones)

        policy_loss = F.cross_entropy(policy, target_policy)
        value_loss = F.mse_loss(value.squeeze(), target_value)
        loss = policy_loss + value_loss

        optimizer = torch.optim.Adam(
            self.neural_net.parameters(),
            lr=training_params["learning_rate"],
            weight_decay=training_params["weight_decay"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.debug(f"Replay loss: {loss.item()}")

    def _compute_targets(self, next_states, rewards, dones):
        config = load_config()
        training_params = config[f"{self.az_id}_training_params"]
        gamma = training_params["gamma"]
        batch_size = next_states.size(0)
        target_policy = torch.zeros(batch_size, self.neural_net.fc_policy.out_features).to(self.neural_net.device)
        target_value = torch.zeros(batch_size).to(self.neural_net.device)

        for i in range(batch_size):
            if dones[i]:
                target_value[i] = rewards[i]
                continue
            checkers_game = CheckersGame()
            state_array = next_states[i].cpu().numpy()
            checkers_game.set_state(state_array)
            move_probs = self.mcts.search(checkers_game)
            for from_idx, to_idx in checkers_game.get_legal_moves():
                idx = from_idx * 32 + to_idx
                if idx < len(move_probs):
                    target_policy[i, idx] = move_probs[idx]
            with torch.no_grad():
                _, next_value = self.neural_net(next_states[i:i + 1])
            target_value[i] = rewards[i] + gamma * next_value.squeeze()

        return target_policy, target_value

    def _idx_to_row_col(self, idx):
        if not (0 <= idx < 32):
            logger.warning(f"Invalid idx: {idx}")
            return None
        row = (idx // 4) * 2 + (1 if (idx % 4) in [0, 1] else 0)
        col = (idx % 4) * 2 + (0 if row % 2 == 0 else 1)
        if not (0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1):
            logger.warning(f"Invalid row,col from idx {idx}: ({row}, {col})")
            return None
        return row, col

    def _row_col_to_idx(self, row, col):
        if not (0 <= row < 8 and 0 <= col < 8 and (row + col) % 2 == 1):
            logger.warning(f"Invalid row,col: ({row}, {col})")
            return None
        idx = ((row // 2) * 4) + ((col // 2) if row % 2 == 0 else (col // 2))
        if not (0 <= idx < 32):
            logger.warning(f"Invalid idx from row,col ({row}, {col}): {idx}")
            return None
        return idx

    def act(self, game, valid_moves):
        logger.debug(f"Entering act: color={self.color}, valid_moves={list(valid_moves.keys())}")
        if not valid_moves:
            logger.warning("No valid moves provided")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            logger.debug(f"Single move available: {move}")
            return move
        self.set_game(game)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        logger.debug(f"Epsilon updated: {self.epsilon}")

        if random.random() < (self.ability / 10.0):
            checkers_game = CheckersGame()
            state = self.get_state(game.board)
            checkers_game.set_state(state)
            checkers_game.current_player = -1 if self.color == "black" else 1
            logger.debug(f"Set current_player={checkers_game.current_player} for color={self.color}")

            logger.debug("Calling MCTS search")
            move_probs = self.mcts.search(checkers_game)
            logger.debug(f"MCTS returned move_probs: shape={move_probs.shape}, sum={np.sum(move_probs)}")

            if move_probs is None or not isinstance(move_probs, np.ndarray):
                logger.error("Invalid move_probs, falling back to uniform distribution")
                move_probs = np.zeros(checkers_game.get_action_size())
                for move in valid_moves:
                    idx = self._row_col_to_idx(*move)
                    if idx is not None:
                        move_probs[idx * 32 + idx] = 1.0 / len(valid_moves)

            legal_indices = []
            for move in valid_moves.keys():
                to_row, to_col = move
                to_idx = self._row_col_to_idx(to_row, to_col)
                if to_idx is None:
                    logger.warning(f"Skipping invalid move: {move}")
                    continue
                for r in range(8):
                    for c in range(8):
                        if (r + c) % 2 == 1:
                            piece = game.board.board[r][c]
                            if piece and piece.is_player_2 == (self.color == "black"):
                                moves = game.get_valid_moves(piece)
                                if move in moves:
                                    from_row, from_col = piece.row, piece.col
                                    from_idx = self._row_col_to_idx(from_row, from_col)
                                    if from_idx is None or to_idx is None:
                                        logger.warning(
                                            f"Invalid indices: from=({from_row}, {from_col}), to=({to_row}, {to_col})")
                                        continue
                                    idx = from_idx * 32 + to_idx
                                    if idx < len(move_probs):
                                        legal_indices.append((idx, (from_row, from_col), move))
                                    else:
                                        logger.warning(
                                            f"Index out of bounds: idx={idx}, move_probs_len={len(move_probs)}")

            checkers_legal_moves = checkers_game.get_legal_moves()
            logger.debug(f"Checkers legal moves: {checkers_legal_moves}")
            filtered_indices = []
            for idx, from_pos, to_pos in legal_indices:
                from_idx = idx // 32
                to_idx = idx % 32
                if (from_idx, to_idx) in checkers_legal_moves:
                    filtered_indices.append((idx, from_pos, to_pos))
                else:
                    logger.warning(f"Move ({from_idx}, {to_idx}) not in checkers_legal_moves")

            if filtered_indices:
                best_idx, best_from, best_move = max(filtered_indices, key=lambda x: move_probs[x[0]])
                logger.debug(f"Selected move: {best_from} -> {best_move}, prob={move_probs[best_idx]}")
                state = self.get_state(game.board)
                next_state = self.get_state(game.board)
                reward = self.reward_calculator.get_reward(game, self.color)
                done = game.game_over
                self.remember(state, best_move, reward, next_state, done)
                return best_move
            else:
                logger.warning("No valid moves in filtered_indices, falling back to random")
                random_move = random.choice(list(valid_moves.keys()))
                logger.debug(f"Random move selected: {random_move}")
                return random_move

        random_move = random.choice(list(valid_moves.keys()))
        logger.debug(f"Random move selected: {random_move}")
        return random_move


class RewardCalculator:
    def get_reward(self, game, color):
        if game.game_over:
            return 1.0 if game.winner == color else -1.0 if game.winner else 0.0
        return 0.0