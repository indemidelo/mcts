import time
from src.MCTS import SimulatedGame
from src.Player import Player
from src.train import Training

start_train = time.time()
p1, p2 = Player(1), Player(2)
for j in range(2):
    print('Game', j + 1)
    count = 0
    mcts = SimulatedGame(p1, p2, n_iter=10, n_moves=20)
    start_time = time.time()
    data_for_training = mcts.play_a_game()
    print(data_for_training.keys())
    print([len(d) for d in data_for_training.values()])
    print(f'Elapsed: {time.time() - start_time}')
print(f'Elapsed train: {time.time() - start_train}')

t = Training(n_games=200, n_iter=10, n_moves=20, n_epochs=10)
t.play_and_train()
