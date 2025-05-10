import json
import os
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from .base_ai import BaseAI
from .config import load_config, load_ai_config
from .rewards import RewardCalculator
from .progress_tracker import ProgressTracker

# متادیتای ماژول
AI_METADATA = {
    "type": "advanced_ai",
    "description": "هوش مصنوعی پیشرفته با شبکه عصبی عمیق و یادگیری تقویتی.",
    "code": "al"
}

project_dir = Path(__file__).parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

parent_dir = project_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

class AdvancedNN(nn.Module):
    def __init__(self, ai_config, player_key):
        super(AdvancedNN, self).__init__()
        # استفاده از تنظیمات از ai_config
        nn_params = ai_config[player_key]["params"]["advanced_nn_params"]
        board_size = ai_config[player_key]["params"]["network_params"]["board_size"]

        self.conv1 = nn.Conv2d(
            nn_params["input_channels"],
            nn_params["conv1_filters"],
            kernel_size=nn_params["conv1_kernel_size"],
            padding=nn_params["conv1_padding"]
        )
        self.residual_block1 = nn.Sequential(
            nn.Conv2d(
                nn_params["residual_block1_filters"],
                nn_params["residual_block1_filters"],
                kernel_size=nn_params["conv1_kernel_size"],
                padding=nn_params["conv1_padding"]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(nn_params["residual_block1_filters"]),
            nn.Conv2d(
                nn_params["residual_block1_filters"],
                nn_params["residual_block1_filters"],
                kernel_size=nn_params["conv1_kernel_size"],
                padding=nn_params["conv1_padding"]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(nn_params["residual_block1_filters"]),
        )
        self.conv2 = nn.Conv2d(
            nn_params["residual_block1_filters"],
            nn_params["conv2_filters"],
            kernel_size=nn_params["conv1_kernel_size"],
            padding=nn_params["conv1_padding"]
        )
        self.residual_block2 = nn.Sequential(
            nn.Conv2d(
                nn_params["residual_block2_filters"],
                nn_params["residual_block2_filters"],
                kernel_size=nn_params["conv1_kernel_size"],
                padding=nn_params["conv1_padding"]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(nn_params["residual_block2_filters"]),
            nn.Conv2d(
                nn_params["residual_block2_filters"],
                nn_params["residual_block2_filters"],
                kernel_size=nn_params["conv1_kernel_size"],
                padding=nn_params["conv1_padding"]
            ),
            nn.ReLU(),
            nn.BatchNorm2d(nn_params["residual_block2_filters"]),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=nn_params["attention_embed_dim"],
            num_heads=nn_params["attention_num_heads"],
            batch_first=True
        )

        fc_input_size = nn_params["residual_block2_filters"] * board_size * board_size
        fc_layers = []
        prev_size = fc_input_size
        for size in nn_params["fc_layer_sizes"]:
            fc_layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(nn_params["dropout_rate"])
            ])
            prev_size = size
        fc_layers.append(nn.Linear(prev_size, board_size * board_size))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.residual = nn.Linear(board_size * board_size * nn_params["input_channels"], board_size * board_size)

    def forward(self, state, ai_config, player_key):
        nn_params = ai_config[player_key]["params"]["advanced_nn_params"]
        board_size = ai_config[player_key]["params"]["network_params"]["board_size"]

        batch_size = state.size(0)
        x = state.view(batch_size, nn_params["input_channels"], board_size, board_size)
        x = F.relu(self.conv1(x))
        x = x + self.residual_block1(x)
        x = F.relu(self.conv2(x))
        x = x + self.residual_block2(x)

        attn_input = x.view(batch_size, board_size * board_size, nn_params["attention_embed_dim"])
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.view(batch_size, nn_params["residual_block2_filters"], board_size, board_size)

        conv_out = attn_output.view(batch_size, -1)
        fc_out = self.fc_layers(conv_out)
        residual_out = self.residual(state.view(batch_size, -1))
        return fc_out + residual_out

class AdvancedAI(BaseAI):
    metadata = AI_METADATA

    def __init__(self, game, model_name, ai_id, config=None):
        super().__init__(game, model_name, ai_id, settings=None)
        config_dict = config if config else {}
        ai_config = config_dict.get("ai_configs", load_ai_config())
        player_key = "player_1" if ai_id == "ai_1" else "player_2"

        self.ability = min(max(int(ai_config[player_key].get("ability_level", 5)), 1), 10)
        self.az_id = ai_id

        pth_dir = Path(__file__).parent.parent / "pth"
        pth_dir.mkdir(exist_ok=True)
        self.model_path = pth_dir / f"{model_name}_{ai_id}.pth"
        self.backup_path = pth_dir / f"backup_{model_name}_{ai_id}.pth"
        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"

        self.training_params = ai_config[player_key]["params"].get("training_params", {})
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = AdvancedNN(ai_config, player_key).to(self.device)
        self.target_net = AdvancedNN(ai_config, player_key).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.training_params.get("learning_rate", 0.0005))

        self.gamma = self.training_params.get("gamma", 0.99)
        self.progress_tracker = ProgressTracker(ai_id)
        self.steps_done = 0
        self.epoch = 0
        self.episode_count = 0
        self.update_target_every = self.training_params.get("update_target_every", 100)
        self.reward_calculator = RewardCalculator(game=game, ai_id=ai_id)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.reward_threshold = self.training_params.get("reward_threshold", 0.5)

        if self.model_path.exists():
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_game(self, game):
        self.game = game
        self.reward_calculator.game = game

    def get_move(self, board):
        print(f"AdvancedAI.get_move called for {self.ai_id} (color: {self.color})")
        valid_moves = self.get_valid_moves(board)
        print(f"Valid moves: {valid_moves}")
        if not valid_moves:
            print("Warning: No valid moves available in AdvancedAI.get_move")
            return None

        move = self.act(valid_moves)
        print(f"Selected move by act: {move}")
        if not move:
            print("Warning: act() returned None, selecting first valid move")
            move = list(valid_moves.keys())[0]

        if isinstance(move, tuple) and len(move) == 2:
            return move
        else:
            print(f"Error: Invalid move format: {move}")
            return None

    def get_valid_moves(self, board):
        valid_moves = {}
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        board_size = ai_config[player_key]["params"]["network_params"]["board_size"]
        print(f"Getting valid moves for {self.ai_id} (color: {self.color}) with board shape: {board.shape}")
        for row in range(board_size):
            for col in range(board_size):
                piece = board[row, col]
                if piece != 0 and ((piece < 0) == (self.color == "black")):
                    print(f"Checking piece at ({row}, {col}): {piece}")
                    moves = self.game.get_valid_moves(row, col)
                    print(f"Moves for ({row}, {col}): {moves}")
                    if moves:
                        for move, skipped in moves.items():
                            valid_moves[((row, col), move)] = skipped
                        print(f"Added moves for ({row}, {col}): {valid_moves}")
        print(f"Final valid moves: {valid_moves}")
        return valid_moves

    def get_state(self, board):
        """تبدیل حالت تخته به فرمت شبکه عصبی."""
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        board_size = ai_config[player_key]["params"]["network_params"]["board_size"]
        nn_params = ai_config[player_key]["params"]["advanced_nn_params"]
        state = np.zeros((nn_params["input_channels"], board_size, board_size))
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
        print(f"State created with shape: {state.shape}")
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def act(self, valid_moves):
        """انتخاب حرکت بر اساس Q-values."""
        print(f"AdvancedAI.act called with valid_moves: {valid_moves}")
        if not valid_moves:
            print("No valid moves in act")
            return None
        if len(valid_moves) == 1:
            move = list(valid_moves.keys())[0]
            print(f"Only one move available: {move}")
            return move

        state = self.get_state(self.game.board.board)
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        print(f"State shape: {state.shape}")
        with torch.no_grad():
            q_values = self.policy_net(state, ai_config, player_key)
            print(f"Q-values shape: {q_values.shape}")
            board_size = ai_config[player_key]["params"]["network_params"]["board_size"]
            valid_q_values = {}
            for (position, move), skipped in valid_moves.items():
                index = move[0] * board_size + move[1]
                try:
                    q_value = q_values[0, index].item()
                    valid_q_values[(position, move)] = q_value
                except IndexError as e:
                    print(f"Index error for move {move}: {e}")
                    continue
            print(f"Valid Q-values: {valid_q_values}")
            if valid_q_values:
                best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
                print(f"Best move: {best_move}")
                return best_move
            print("Warning: No valid Q-values, selecting first valid move")
            return list(valid_moves.keys())[0]

    def update(self, move, reward):
        """به‌روزرسانی مدل با تجربه جدید."""
        state = self.get_state(self.game.board.board)
        action = move[1] if move else None
        next_state = self.get_state(self.game.board.board)
        done = self.game.game_over

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
        ai_config = load_ai_config()
        player_key = "player_1" if self.ai_id == "ai_1" else "player_2"
        training_params = ai_config[player_key]["params"]["training_params"]
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
        board_size = ai_config[player_key]["params"]["network_params"]["board_size"]
        actions = torch.tensor([(a[0] * board_size + a[1]) for a in actions if a], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states, ai_config, player_key).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states, ai_config, player_key).max(1)[0]
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