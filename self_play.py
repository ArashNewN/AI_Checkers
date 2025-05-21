import numpy as np
from collections import deque
import torch
from mcts import MCTS  # توجه به import نسبی
import logging
import json
import time
from pathlib import Path
from config import load_config  # بارگذاری تنظیمات از config.py

# بارگذاری تنظیمات
config = load_config()

# تنظیم مسیر برای ذخیره حرکات بازی
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
move_history_dir = parent_dir / "move_history"
move_history_dir.mkdir(parents=True, exist_ok=True)  # ایجاد پوشه اگر وجود ندارد

# تنظیم لاگ‌گیری
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('self_play.log')]
)
logger = logging.getLogger(__name__)

def self_play(game, neural_net, start_game=0):
    """
    اجرای بازی به روش Self-Play
    ذخیره حرکات بازی و به‌روزرسانی حافظه برای آموزش هوش مصنوعی.
    """
    memory = deque(maxlen=config["memory_size"])  # گرفتن مقدار از config
    for game_idx in range(start_game, config["num_games"]):  # گرفتن تعداد بازی از config
        state = game.copy().reset()
        game_history = []  # ذخیره حرکات بازی
        move_count = 0
        outcome = 0

        while move_count < config["max_moves"]:  # گرفتن حداکثر حرکت‌ها از config
            if state.is_terminal():  # بررسی پایان بازی
                outcome = state.get_outcome()
                break

            # جستجوی MCTS
            start_time = time.time()
            mcts = MCTS(game, neural_net)
            move_probs = mcts.search(state)
            logger.debug(f"MCTS search took {time.time() - start_time:.2f} seconds")

            legal_moves = state.get_legal_moves()
            if not legal_moves:  # بررسی حرکات قانونی
                outcome = state.get_outcome()
                break

            valid_probs = np.zeros(len(move_probs))
            for move in legal_moves:
                from_idx, to_idx = move
                idx = from_idx * 32 + to_idx
                valid_probs[idx] = move_probs[idx]
            valid_probs /= np.sum(valid_probs) if np.sum(valid_probs) > 0 else 1

            move_idx = np.random.choice(len(move_probs), p=valid_probs)
            move = (move_idx // 32, move_idx % 32)
            if move not in legal_moves:
                logger.debug(f"Selected move {move} not in legal moves, choosing random legal move")
                move = legal_moves[np.random.randint(len(legal_moves))]

            current_state = state.get_state()
            game_history.append({
                "move_count": move_count + 1,
                "player": state.current_player,
                "move": move,
                "board_state": current_state.tolist()  # ذخیره وضعیت تخته به‌صورت لیست
            })

            try:
                state = state.make_move(move)
            except ValueError as e:
                logger.error(f"Error applying move {move}: {e}")
                outcome = state.get_outcome()
                break

            move_count += 1
            outcome = state.get_outcome()
            if outcome != 0:
                break

        # ذخیره حرکات این بازی در فایل JSON
        game_file = move_history_dir / f"game_{game_idx + 1}.json"
        with open(game_file, "w", encoding="utf-8") as f:
            json.dump(game_history, f, ensure_ascii=False, indent=4)
        logger.info(f"Game {game_idx + 1} moves saved to {game_file}")

        # به‌روزرسانی حافظه برای آموزش
        for entry in game_history:
            state = entry["board_state"]
            probs = entry["move"]
            player = entry["player"]
            reward = outcome if player == 1 else -outcome
            memory.append((state, probs, reward))
        logger.info(f"Game {game_idx + 1} finished with outcome: {outcome}")
        logger.info(f"Game {game_idx + 1} took {move_count} moves")

        # پاک‌سازی فایل‌های قدیمی برای محدود کردن تعداد بازی‌های ذخیره‌شده
        clean_old_games(config["max_history_files"])

    return memory

def clean_old_games(max_games=100):
    """حذف فایل‌های قدیمی برای محدود کردن تعداد بازی‌های ذخیره‌شده"""
    game_files = sorted(move_history_dir.glob("game_*.json"))
    if len(game_files) > max_games:
        old_files = game_files[:-max_games]
        for old_file in old_files:
            old_file.unlink()
        logger.info(f"Removed {len(old_files)} old game files.")

def replay_game(game_file):
    """بازپخش یک بازی ذخیره‌شده"""
    with open(game_file, "r", encoding="utf-8") as f:
        moves = json.load(f)

    for move in moves:
        print(f"Move {move['move_count']} - Player {move['player']} moved {move['move']}")
        print(f"Board State:\n{np.array(move['board_state'])}")

def replay_all_games():
    """بازپخش تمام بازی‌های ذخیره‌شده"""
    game_files = sorted(move_history_dir.glob("game_*.json"))
    for game_file in game_files:
        print(f"Replaying {game_file.name}")
        replay_game(game_file)