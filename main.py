from src.train import Training

t = Training(n_games=10, n_iter=10, n_moves=10, n_epochs=5)
t.play_and_train()
