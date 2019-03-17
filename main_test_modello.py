from src.train import Training

if __name__ == '__main__':
    # Training
    print('Neural Network Training')
    t = Training(n_games=25, n_iter=50, n_moves=40, n_epochs=5)
    t.play_and_train()

    # Testing
    print('Neural Network Testing')
    t.test()
