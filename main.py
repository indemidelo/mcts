from src.train import Training

t = Training(n_games=2, n_iter=50, n_moves=30, n_epochs=5, batch_size=100)
# t = Training(n_games=500, n_iter=500, n_moves=300, n_epochs=25, batch_size=100)
t.train()