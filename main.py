from src.MCTS import MCTS

mcts = MCTS(10)
mcts.initialize()

count = 0
while mcts.board.playing or count > 500:
    mcts.loop()
    print(mcts.board)
    count += 1
