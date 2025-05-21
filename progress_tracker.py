import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.INFO,
    format=' %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('progress_tracker.log')]
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self, ai_id):
        self.al_id = ai_id  # ذخیره al_id برای شناسایی AI خاص
        # تعیین مسیر پوشه log
        current_dir = os.path.abspath(os.getcwd())  # مسیر فعلی
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))  # پوشه بالاتر
        base_log_dir = os.path.join(parent_dir, "train log png")
        os.makedirs(base_log_dir, exist_ok=True)

        # ایجاد پوشه با نام تاریخ و زمان و al_id
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(base_log_dir, f"{self.al_id}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # تنظیم مسیر فایل‌های متنی و نمودارها با استفاده از al_id
        self.progress_file = os.path.join(self.log_dir, f"training_progress_{self.al_id}.txt")
        self.plot_file = os.path.join(self.log_dir, f"training_progress_{self.al_id}.png")
        
        # داده‌های ذخیره‌ای
        self.epochs = []
        self.policy_losses = []
        self.value_losses = []
        self.player1_wins = []
        self.player2_wins = []

    def log_epoch(self, epoch, policy_loss, value_loss, player1_wins, player2_wins):
        """ثبت اطلاعات یک دوره"""
        self.epochs.append(epoch)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.player1_wins.append(player1_wins)
        self.player2_wins.append(player2_wins)

        # ذخیره اطلاعات در فایل متنی
        with open(self.progress_file, "a") as f:
            f.write(
                f"Epoch {epoch}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
                f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}\n"
            )
        logger.info(
            f"[{self.al_id}] Epoch {epoch}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, "
            f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}"
        )

    def plot_progress(self):
        """رسم و ذخیره نمودار پیشرفت"""
        plt.figure(figsize=(12, 8))

        # نمودار Loss
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.policy_losses, label="Policy Loss", marker='o')
        plt.plot(self.epochs, self.value_losses, label="Value Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Over Epochs [{self.al_id}]")
        plt.legend()
        plt.grid(True)

        # نمودار برد بازیکنان
        plt.subplot(2, 1, 2)
        plt.plot(self.epochs, self.player1_wins, label="Player 1 Wins", marker='o')
        plt.plot(self.epochs, self.player2_wins, label="Player 2 Wins", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Number of Wins")
        plt.title(f"Player Wins Over Epochs [{self.al_id}]")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.plot_file)
        logger.info(f"[{self.al_id}] Progress plot saved to {self.plot_file}")

    def save_summary(self):
        """ذخیره خلاصه پیشرفت در فایل متنی"""
        summary_file = os.path.join(self.log_dir, f"training_summary_{self.al_id}.txt")
        with open(summary_file, "w") as f:
            f.write(f"Training Progress Summary [{self.al_id}]\n")
            f.write("======================\n")
            for i, epoch in enumerate(self.epochs):
                f.write(
                    f"Epoch {epoch}, Policy Loss: {self.policy_losses[i]:.4f}, "
                    f"Value Loss: {self.value_losses[i]:.4f}, "
                    f"Player 1 Wins: {self.player1_wins[i]}, Player 2 Wins: {self.player2_wins[i]}\n"
                )
        logger.info(f"[{self.al_id}] Training summary saved to {summary_file}")