import time
from src.MCTS import SimulatedGame
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


start_train = time.time()
nn = NeuralNetwork()
for j in range(20):
    print('Game', j+1)
    count = 0
    mcts = SimulatedGame(Player(1), Player(2), nn, 10, 20)
    mcts.initialize()
    start_time = time.time()
    while mcts.board.playing or count > 500:
        mcts.move()
        # print(mcts.board)
        count += 1
    print(f'Elapsed: {time.time() - start_time}')
print(f'Elapsed train: {time.time() - start_train}')
