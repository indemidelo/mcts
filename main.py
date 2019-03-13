import time
from src.MCTS import SimulatedGame
from src.Logger import Logger
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


start_train = time.time()
nn = NeuralNetwork()
logger = Logger()
for j in range(2):
    print('Game', j+1)
    count = 0
    mcts = SimulatedGame(
        Player(1), Player(2), nn, n_iter=10,
        n_moves=20, logger=logger)
    start_time = time.time()
    data_for_training = mcts.play_a_game()
    print(data_for_training)
    print(f'Elapsed: {time.time() - start_time}')
print(f'Elapsed train: {time.time() - start_train}')
