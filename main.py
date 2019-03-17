from src.train import Training

t = Training(n_games=10, n_iter=500, n_moves=20, n_epochs=5)
t.play_and_train()