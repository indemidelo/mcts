from src.train import Training

if __name__ == '__main__':
    # Training
    print('Neural Network Training')

    # with these hyperparameters it learns how to win with stacks of fishes
    # t = Training(n_games=2, n_iter=5, n_moves=4, n_epochs=2, batch_size=100)

    t = Training()
    t.train()

    # Testing
    print('Neural Network Testing')
    t.test()
