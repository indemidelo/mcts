from src.train import Training
from config import CFG

if __name__ == '__main__':
    # Training
    print('Neural Network Training')
    if CFG.game == 1:
        from games.tic_tac_toe.TicTacToe import TicTacToe
        game = TicTacToe
    else:
        from games.connect_four.ConnectFour import ConnectFour
        game = ConnectFour
    t = Training(game)
    t.train()

    # Testing
    print('Neural Network Testing')
    t.test()
