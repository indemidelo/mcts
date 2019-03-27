from src.train import Training

if __name__ == '__main__':
    # Training
    print('Neural Network Training')
    t = Training()
    t.train()

    # Testing
    print('Neural Network Testing')
    t.test_ai()
