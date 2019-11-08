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


    if CFG.mode == 'nn_train':
        t = Training(game)
        t.train()

    elif CFG.mode == 'nn_train_test':
        # Testing
        t = Training(game)
        t.train()
        print('Neural Network Testing')
        t.test()

    elif CFG.mode == 'pure_rl_test':
        from src.MCTS import SimulatedGame
        from src.pure_rl_agent.PureRLAgent import PureRLAgent
        agent = PureRLAgent(game)
        simgame = SimulatedGame(agent)
        simgame.play_a_game(print_board=True)
