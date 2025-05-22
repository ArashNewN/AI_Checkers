import os
import torch
import torch.optim as optim
import logging
import numpy as np
import pickle
from collections import deque
from pathlib import Path

from .alphazero_net import AlphaZeroNet
from .self_play import self_play
from .progress_tracker import ProgressTracker
from pyfile.checkers_game import CheckersGame
from pyfile.config import load_config  # بارگذاری تنظیمات از فایل config.json


# بارگذاری تنظیمات
config = load_config()

# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('train.log')]
)
logger = logging.getLogger(__name__)

def train_alphazero(game):
        # ساخت مدل AlphaZero
    model = AlphaZeroNet(game).to(device)
    # مقادیر تنظیم‌شده در فایل config.json
    num_episodes = config["num_episodes"]
    num_games = config["num_games"]
    mcts_simulations = config["mcts_simulations"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    max_moves = config["max_moves"]

    # ساخت پوشه‌ها برای مدل و چک‌پوینت‌ها
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent

    # پوشه برای ذخیره فایل‌های مدل
    pth_dir = parent_dir / "pth"
    os.makedirs(pth_dir, exist_ok=True)
    model_path = pth_dir / "alphazero_model.pth"

    # مسیر فایل‌های چک‌پوینت
    epoch_dir = parent_dir / "epoch"
    os.makedirs(epoch_dir, exist_ok=True)
    checkpoint_path = epoch_dir / "checkpoint.pkl"

    # مسیر فایل‌های مدل
    pth_dir = parent_dir / "pth"
    os.makedirs(pth_dir, exist_ok=True)
    model_path = pth_dir / "alphazero_model.pth"

    # بک‌آپ گرفتن از فایل مدل قبلی
    if model_path.exists():
        backup_path = pth_dir / f"{model_path.stem}.backup{model_path.suffix}"
        os.rename(model_path, backup_path)
        logger.info(f"Backed up existing model to {backup_path}")

    else:
        logger.info(f"No existing model found at {model_path}. Starting with new model.")
    
    # لود کردن مدل موجود
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path))
            logger.info(f"Loaded existing model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}. Starting with new model.")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    progress_tracker = ProgressTracker()

    # لود چک‌پوینت اگر وجود داشته باشد
    start_epoch = 0
    memory = deque(maxlen=config["memory_size"])
    if checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        start_epoch = checkpoint['epoch']
        memory = checkpoint['memory']
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info(f"Resumed from checkpoint: epoch {start_epoch}")

    for epoch in range(start_epoch, num_episodes):
        logger.info(f"Starting episode {epoch + 1}/{num_episodes}")
        memory = self_play(game, model, num_games=num_games, max_moves=max_moves, mcts_simulations=mcts_simulations)
        
        # لاگ کردن تعداد نمونه‌ها
        logger.info(f"Collected {len(memory)} new samples. Total memory size: {len(memory)}")
        
        # شمارش تعداد بردها
        player1_wins = sum(1 for state, probs, reward in memory if reward == 1)
        player2_wins = sum(1 for state, probs, reward in memory if reward == -1)
        
        # آموزش شبکه عصبی
        model.train()
        total_policy_loss = 0
        total_value_loss = 0
        for i in range(0, len(memory), batch_size):
            indices = np.random.choice(len(memory), size=min(batch_size, len(memory) - i), replace=False)
            batch = [memory[idx] for idx in indices]
            states, move_probs, rewards = zip(*batch)
            
            states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            move_probs = torch.tensor(np.array(move_probs), dtype=torch.float32).to(device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            policy, value = model(states)
            policy_loss = -torch.mean(torch.sum(move_probs * torch.log(policy + 1e-10), dim=1))
            value_loss = torch.mean((rewards - value.squeeze()) ** 2)
            total_loss = policy_loss + value_loss
            total_loss.backward()
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # ذخیره چک‌پوینت هر 10 بازی
            if (i // batch_size + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'memory': memory,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        avg_policy_loss = total_policy_loss / max(1, (len(memory) / batch_size))
        avg_value_loss = total_value_loss / max(1, (len(memory) / batch_size))
        
        # لاگ کردن پیشرفت
        progress_tracker.log_epoch(epoch + 1, avg_policy_loss, avg_value_loss, player1_wins, player2_wins)
        
        # ذخیره مدل به ازای هر دوره
        epoch_model_path = pth_dir / f"alphazero_model_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(f"Saved model to {epoch_model_path}")
        
        # ذخیره مدل اصلی
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved main model to {model_path}")
        
        # رسم نمودار پیشرفت
        progress_tracker.plot_progress()
    
    # ذخیره خلاصه نهایی
    progress_tracker.save_summary()
    
    # حذف چک‌پوینت بعد از اتمام
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info(f"Removed checkpoint file {checkpoint_path}")

    if backup_path.exists():
        backup_path.unlink()
        logger.info(f"Removed backup file {backup_path}")

if __name__ == "__main__":
    game = CheckersGame()
    train_alphazero(game)