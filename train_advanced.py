# train.py
import torch
from .alphazero_ai import AlphaZeroAI, train_alphazero
from .checkers_game import CheckersGame

game = CheckersGame()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_alphazero(
    game,
    model_path="alphazero_model_p1.pth",
    num_episodes=1000,
    num_games=100,
    mcts_simulations=800,
    device=device
)
train_alphazero(
    game,
    model_path="alphazero_model_p2.pth",
    num_episodes=1000,
    num_games=100,
    mcts_simulations=800,
    device=device
)