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
from .rewards import RewardCalculator
from .base_ai import BaseAI

AI_METADATA = {
    "type": "alhazero_ai",
    "description": "هوش مصنوعی پیشرفته ...",
    "code": "az"
}

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

class AlphaZeroAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, color, model_name, ai_id, settings=None):
        super().__init__(game, color, model_name, ai_id, settings)
        config = load_config()
        ai_config = load_ai_config()  # بارگذاری ai_config.json
        player_key = "player_1" if ai_id == "ai_1" else "player_2"

        # خواندن ability از ai_config.json
        self.ability = min(max(int(ai_config["ai_configs"][player_key].get("ability_level", 5)), 1), 10)
        self.az_id = ai_id

        # تعریف مسیرهای ذخیره مدل
        pth_dir = Path(__file__).parent.parent / "pth"
        pth_dir.mkdir(exist_ok=True)
        self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
        self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"

        # پارامترهای آموزش
        self.training_params = config.get(f"{ai_id}_training_params", {
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "replay_batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "gamma": 0.99,
            "memory_size": 10000,
            "batch_size": 128,
            "update_target_every": 100,
            "reward_threshold": 0.5
        })
        self.epsilon = self.training_params["epsilon_start"]
        self.epsilon_end = self.training_params["epsilon_end"]
        self.epsilon_decay = self.training_params["epsilon_decay"]
        self.gamma = self.training_params["gamma"]
        self.reward_threshold = self.training_params["reward_threshold"]

        # حافظه و دستگاه
        self.memory = deque(maxlen=self.training_params["memory_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # شبکه‌های عصبی
        self.policy_net = AlphaZeroNet().to(self.device)
        self.target_net = AlphaZeroNet().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.training_params["learning_rate"])

        # متغیرهای آموزش
        self.progress_tracker = ProgressTracker(ai_id)
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.training_params["update_target_every"]
        self.episode_rewards = []
        self.current_episode_reward = 0

        # MCTS و بازی
        self.game = game  # همیشه از game پاس‌شده استفاده کن
        self.mcts = MCTS(self, CheckersGame(), self.policy_net)
        self.reward_calculator = RewardCalculator(game=self.game, ai_id=ai_id)

        # بارگذاری مدل‌های قبلی
        if self.model_path.exists():
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

        logger.debug(f"Initialized AlphaZeroAI with az_id={self.az_id}, ability={self.ability}")

    def set_game(self, game):
        self.game = game
        self.reward_calculator.game = game

    def get_state(self, board):
        config = load_config()
        board_size = config["network_params"]["board_size"]
        state = np.zeros((4, board_size, board_size), dtype=np.float32)
        for row in range(board_size):
            for col in range(board_size):
                piece = board[row, col]
                if piece != 0:
                    is_player_2 = piece < 0
                    is_king = abs(piece) == 2
                    if is_player_2:
                        state[2, row, col] = 1.0 if not is_king else 0.0
                        state[3, row, col] = 1.0 if is_king else 0.0
                    else:
                        state[0, row, col] = 1.0 if not is_king else 0.0
                        state[1, row, col] = 1.0 if is_king else 0.0
        logger.debug(f"State created with shape: {state.shape}")
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def convert_state_to_network_format(self, state):
        return state

    def act(self, valid_moves):
        logger.debug(f"Entering act: color={self.color}, valid_moves={list(valid_moves.keys())}")
        if not valid_moves:
            logger.warning("No valid moves provided")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            logger.debug(f"Single move available: {move}")
            return move

        # آماده‌سازی CheckersGame برای MCTS
        checkers_game = CheckersGame()
        state = self.get_state(self.game.board.board)
        checkers_game.set_state(state.cpu().numpy())  # تبدیل به NumPy
        checkers_game.current_player = -1 if self.color == "black" else 1
        logger.debug(f"Set current_player={checkers_game.current_player} for color={self.color}")

        logger.debug("Calling MCTS search")
        move_probs = self.mcts.search(checkers_game)
        logger.debug(f"MCTS returned move_probs: shape={move_probs.shape}, sum={np.sum(move_probs)}")

        if move_probs is None or not isinstance(move_probs, np.ndarray):
            logger.error("Invalid move_probs returned by MCTS")
            return random.choice(list(valid_moves.keys()))

        # انتخاب بهترین حرکت از valid_moves
        best_move = None
        best_prob = -1
        for move in valid_moves.keys():
            from_row, from_col = move[0]
            to_row, to_col = move[1]
            from_idx = self._row_col_to_idx(from_row, from_col)
            to_idx = self._row_col_to_idx(to_row, to_col)
            if from_idx is None or to_idx is None:
                logger.warning(f"Invalid indices: from=({from_row}, {from_col}), to=({to_row}, {to_col})")
                continue
            idx = from_idx * 32 + to_idx
            if idx < len(move_probs):
                piece = self.game.board.board[from_row, from_col]
                if piece != 0 and (piece < 0) == (self.color == "black"):
                    prob = move_probs[idx]
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move
                else:
                    logger.debug(f"Invalid piece at ({from_row}, {from_col}): {piece}")
            else:
                logger.warning(f"Index out of bounds: idx={idx}, move_probs_len={len(move_probs)}")

        if best_move:
            logger.debug(f"Selected move: {best_move}, prob={best_prob}")
            state = self.get_state(self.game.board.board)
            next_state = self.get_state(self.game.board.board)
            reward = self.reward_calculator.get_reward()
            done = self.game.game_over
            self.remember(state, best_move, reward, next_state, done)
            return best_move
        else:
            logger.warning("No valid moves found after filtering")
            move = random.choice(list(valid_moves.keys()))
            logger.debug(f"Fallback to random move: {move}")
            return move

    def act(self, valid_moves):
        logger.debug(f"Entering act: color={self.color}, valid_moves={list(valid_moves.keys())}")
        if not valid_moves:
            logger.warning("No valid moves provided")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            logger.debug(f"Single move available: {move}")
            return move

        # آماده‌سازی CheckersGame
        checkers_game = CheckersGame()
        state = self.get_state(self.game.board.board)
        checkers_game.set_state(state.cpu().numpy())
        checkers_game.current_player = -1 if self.color == "black" else 1
        logger.debug(f"Set current_player={checkers_game.current_player} for color={self.color}")

        # اجرای MCTS
        logger.debug("Calling MCTS search")
        move_probs = self.mcts.search(checkers_game)
        logger.debug(f"MCTS returned move_probs: shape={move_probs.shape}, sum={np.sum(move_probs)}")

        if move_probs is None or not isinstance(move_probs, np.ndarray):
            logger.error("Invalid move_probs returned by MCTS")
            return random.choice(list(valid_moves.keys()))

        # تبدیل حرکت‌های CheckersGame به فرمت valid_moves
        checkers_legal_moves = checkers_game.get_legal_moves()  # حالا به شکل ((from_row, from_col), (to_row, to_col))
        best_move = None
        best_prob = -1
        for move in valid_moves.keys():
            if move in checkers_legal_moves:  # مقایسه مستقیم
                from_row, from_col = move[0]
                to_row, to_col = move[1]
                from_idx = self._row_col_to_idx(from_row, from_col)
                to_idx = self._row_col_to_idx(to_row, to_col)
                if from_idx is None or to_idx is None:
                    logger.warning(f"Invalid indices: from=({from_row}, {from_col}), to=({to_row}, {to_col})")
                    continue
                idx = from_idx * 32 + to_idx
                if idx < len(move_probs):
                    prob = move_probs[idx]
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move
                else:
                    logger.warning(f"Index out of bounds: idx={idx}, move_probs_len={len(move_probs)}")

        if best_move:
            logger.debug(f"Selected move: {best_move}, prob={best_prob}")
            state = self.get_state(self.game.board.board)
            next_state = self.get_state(self.game.board.board)
            reward = self.reward_calculator.get_reward()
            done = self.game.game_over
            self.remember(state, best_move, reward, next_state, done)
            return best_move
        else:
            logger.warning("No valid moves found after filtering")
            move = random.choice(list(valid_moves.keys()))
            logger.debug(f"Fallback to random move: {move}")
            return move

    def update(self, move, reward):
        """به‌روزرسانی مدل با تجربه جدید."""
        state = self.get_state(self.game.board.board)  # استفاده از board.board
        action = move[1] if move else None
        next_state = self.get_state(self.game.board.board)
        done = self.game.game_over  # استفاده از ویژگی game_over

        self.remember(state, action, next_state, done)
        self.replay()
        self.update_target_network()
        self.save_model()

    def remember(self, state, action, next_state, done):
        reward = self.reward_calculator.get_reward()
        self.current_episode_reward += reward
        self.memory.append((state, action, reward, next_state, done))
        self.save_important_experience(state, action, reward, next_state, done)
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
            with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
                json.dump(self.episode_rewards, f)

    def replay(self):
        config = load_config()
        training_params = config[f"{self.ai_id}_training_params"]
        batch_size = training_params.get("batch_size", 128)
        if len(self.memory) < batch_size:
            return

        batch = list(self.memory)[-batch_size:]
        long_term_memory = self.load_long_term_memory()
        if long_term_memory:
            long_term_batch = sorted(long_term_memory, key=lambda x: x[2], reverse=True)[:batch_size // 2]
            batch = batch[:(batch_size // 2)] + long_term_batch

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        board_size = config["network_params"]["board_size"]
        actions = torch.tensor([(a[0] * board_size + a[1]) for a in actions if a], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), training_params.get("gradient_clip", 1.0))
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                (1 - training_params.get("target_update_alpha", 0.01)) * target_param.data +
                training_params.get("target_update_alpha", 0.01) * policy_param.data
            )

        self.epoch += 1
        policy_loss = loss.item()
        player1_wins = sum(1 for r in self.episode_rewards if r > 0)
        player2_wins = sum(1 for r in self.episode_rewards if r < 0)
        self.progress_tracker.log_epoch(self.epoch, policy_loss, policy_loss, player1_wins, player2_wins)

    def _compute_targets(self, next_states, rewards, dones):
        config = load_config()
        training_params = config.get(f"{self.az_id}_training_params", {"gamma": 0.99})
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

    def save_important_experience(self, state, action, reward, next_state, done):
        if reward > self.reward_threshold:
            with open(self.long_term_memory_path, "a") as f:
                experience = {
                    "state": state.cpu().numpy().tolist(),
                    "action": action,
                    "reward": float(reward),
                    "next_state": next_state.cpu().numpy().tolist(),
                    "done": done,
                }
                f.write(json.dumps(experience) + "\n")

    def load_long_term_memory(self):
        if os.path.exists(self.long_term_memory_path):
            with open(self.long_term_memory_path, "r") as f:
                experiences = []
                for line in f.readlines():
                    exp = json.loads(line)
                    exp["state"] = torch.FloatTensor(exp["state"]).to(self.device)
                    exp["next_state"] = torch.FloatTensor(exp["next_state"]).to(self.device)
                    experiences.append((
                        exp["state"],
                        exp["action"],
                        exp["reward"],
                        exp["next_state"],
                        exp["done"]
                    ))
                return experiences
        return []

    def update_target_network(self):
        self.episode_count += 1
        if self.episode_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)
        torch.save(self.policy_net.state_dict(), self.backup_path)
        with open(self.long_term_memory_path.parent / f"episode_rewards_{self.ai_id}.json", "w") as f:
            json.dump(self.episode_rewards, f)