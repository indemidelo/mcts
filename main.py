from src.train import Training

t = Training(n_games=100, n_iter=50, n_moves=30, n_epochs=5)
t.play_and_train()