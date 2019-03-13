from src.MCTS import MCTS
import time


start_train = time.time()
for j in range(20):
    print('Game', j+1)
    count = 0
    mcts = MCTS(200)
    mcts.initialize()
    start_time = time.time()
    while mcts.board.playing or count > 500:
        mcts.move()
        # print(mcts.board)
        count += 1
    print(f'Elapsed: {time.time() - start_time}')
print(f'Elapsed train: {time.time() - start_train}')
