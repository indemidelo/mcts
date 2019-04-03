from src.train import Training
from games.connect_four.ConnectFour import ConnectFour

if __name__ == '__main__':
    # Training
    print('Neural Network Training')
    game = ConnectFour
    t = Training(game)
    t.train()

    # Testing
    print('Neural Network Testing')
    t.test()
