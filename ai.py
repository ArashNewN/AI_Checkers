import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from collections import deque
from .config import load_config
from .rewards import RewardCalculator
from pathlib import Path

class AdvancedNN(nn.Module):
    def __init__(self):
        super(AdvancedNN, self).__init__()
        config = load_config()
        nn_params = config["advanced_nn_params"]
        board_size = config["network_params"]["board_size"]

        # لایه‌های کانولوشنی
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

        # لایه توجه
        self.attention = nn.MultiheadAttention(
            embed_dim=nn_params["attention_embed_dim"],
            num_heads=nn_params["attention_num_heads"],
            batch_first=True
        )

        # لایه‌های خطی
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

        # لایه رزیدوال
        self.residual = nn.Linear(board_size * board_size * nn_params["input_channels"], board_size * board_size)

    def forward(self, state):
        config = load_config()
        board_size = config["network_params"]["board_size"]
        nn_params = config["advanced_nn_params"]

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

class AdvancedAI:
    def __init__(self, player_color, model_name="ai_model", ai_id="ai_1"):
        config = load_config()
        training_params = config[f"{ai_id}_training_params"]
        # انتخاب ability بر اساس ai_id
        self.ability = min(max(int(config.get(f"{ai_id}_ability", 5)), 1), 10)
        self.player_color = player_color
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = AdvancedNN().to(self.device)
        self.target_net = AdvancedNN().to(self.device)

        # بارگذاری پارامترهای آموزش
        self.memory_size = training_params["memory_size"]
        self.batch_size = training_params["batch_size"]
        self.learning_rate = training_params["learning_rate"]
        self.gamma = training_params["gamma"]
        self.epsilon_start = training_params["epsilon_start"]
        self.epsilon_end = training_params["epsilon_end"]
        self.epsilon_decay = training_params["epsilon_decay"]
        self.update_target_every = training_params["update_target_every"]
        self.reward_threshold = training_params["reward_threshold"]
        self.gradient_clip = training_params["gradient_clip"]
        self.target_update_alpha = training_params["target_update_alpha"]

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        self.epsilon = self.epsilon_start
        self.steps_done = 0
        self.episode_count = 0

        # تنظیم مسیرهای مدل و بکاپ
        parent_dir = Path(__file__).parent.parent
        pth_dir = parent_dir / "pth"
        if not pth_dir.exists():
            pth_dir.mkdir()
        self.model_path = pth_dir / f"{model_name}.pth"
        self.backup_path = pth_dir / f"{model_name}_backup.pth"

        # مقداردهی اولیه یا بارگذاری مدل
        if not self.model_path.exists():
            torch.save(self.policy_net.state_dict(), self.model_path)
            if not self.backup_path.exists():
                torch.save(self.policy_net.state_dict(), self.backup_path)
        else:
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path))
                self.target_net.load_state_dict(torch.load(self.model_path))
                # به‌روزرسانی بکاپ
                torch.save(self.policy_net.state_dict(), self.backup_path)
            except (RuntimeError, FileNotFoundError, EOFError):
                print(f"Corrupted model file at {self.model_path}, attempting to use backup")
                if self.backup_path.exists():
                    try:
                        self.policy_net.load_state_dict(torch.load(self.backup_path))
                        self.target_net.load_state_dict(torch.load(self.backup_path))
                        torch.save(self.policy_net.state_dict(), self.model_path)
                    except (RuntimeError, FileNotFoundError, EOFError):
                        print(f"Backup file corrupted, initializing new model")
                        torch.save(self.policy_net.state_dict(), self.model_path)
                        torch.save(self.policy_net.state_dict(), self.backup_path)
                else:
                    print(f"No backup available, initializing new model")
                    torch.save(self.policy_net.state_dict(), self.model_path)
                    torch.save(self.policy_net.state_dict(), self.backup_path)

        self.long_term_memory_path = pth_dir / f"long_term_memory_{ai_id}.json"
        self.reward_calculator = RewardCalculator(None)  # بعداً با set_game تنظیم می‌شود

    def set_game(self, game):
        """تنظیم نمونه بازی برای محاسبه پاداش"""
        self.reward_calculator = RewardCalculator(game)

    def get_state(self, board):
        config = load_config()
        board_size = config["network_params"]["board_size"]
        nn_params = config["advanced_nn_params"]
        state = np.zeros((nn_params["input_channels"], board_size, board_size))
        for row in range(board_size):
            for col in range(board_size):
                piece = board.board[row][col]
                if piece:
                    channel = 0 if piece.is_player_2 else 1
                    state[channel, row, col] = 1
                    if piece.king:
                        state[2, row, col] = 1
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def remember(self, state, action, next_state, done):
        reward = self.reward_calculator.get_reward()
        self.memory.append((state, action, reward, next_state, done))
        self.save_important_experience(state, action, reward, next_state, done)

    def act(self, game, valid_moves):
        if not valid_moves:
            return None
        state = self.get_state(game.board)
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        if random.random() < (self.ability / 10.0):
            with torch.no_grad():
                q_values = self.policy_net(state)
                config = load_config()
                board_size = config["network_params"]["board_size"]
                valid_q_values = {move: q_values[0, move[0] * board_size + move[1]].item() for move in valid_moves.keys()}
                return max(valid_q_values.items(), key=lambda x: x[1])[0]
        return random.choice(list(valid_moves.keys()))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        short_term_batch = random.sample(self.memory, self.batch_size // 2)
        long_term_memory = self.load_long_term_memory()
        if long_term_memory:
            long_term_batch = random.sample(long_term_memory, self.batch_size // 2)
        else:
            long_term_batch = []

        combined_batch = short_term_batch + long_term_batch
        states, actions, rewards, next_states, dones = zip(*combined_batch)

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        config = load_config()
        board_size = config["network_params"]["board_size"]
        actions = torch.tensor([(a[0] * board_size + a[1]) for a in actions], device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_((1 - self.target_update_alpha) * target_param.data + self.target_update_alpha * policy_param.data)

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

    def suggest_move(self, game, valid_moves):
        if not valid_moves:
            return None
        state = self.get_state(game.board)
        with torch.no_grad():
            q_values = self.policy_net(state)
            config = load_config()
            board_size = config["network_params"]["board_size"]
            valid_q_values = {move: q_values[0, move[0] * board_size + move[1]].item() for move in valid_moves.keys()}
            best_move = max(valid_q_values.items(), key=lambda x: x[1])[0]
            return best_move