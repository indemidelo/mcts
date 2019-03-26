import numpy as np
from src.config import CFG
from src.MCTS import SimulatedGame
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


class Training():
    def __init__(self):
        self.nn = NeuralNetwork()
        self.p1, self.p2 = Player(1), Player(2)

    def train(self):

        for i in range(CFG.num_iterations):

            training_data = {'state': [], 'pi': [], 'z': []}

            for g in range(CFG.num_games):
                simgame = SimulatedGame(g + i * CFG.num_iterations)
                training_data_loop = simgame.play_a_game()
                self.update_training_data_(training_data, training_data_loop)
                print(f'Game {g + 1} in iter {i + 1} won by player {simgame.tree.board.winner}')

            self.nn.train(*self.prepare_data(training_data))

            if (i + 1) % CFG.checkpoint == 0:
                self.test()
                self.nn.save(i + 1)

    def test(self):
        from src.Board import Board
        from src.tfPlayer import tfPlayer
        from src.NNGame import NNRecordedGame
        b = Board()
        p1 = tfPlayer(1, b, self.nn.sess, self.nn.pred_policy,
                      self.nn.inputs, training=False)
        p2 = tfPlayer(-1, b, self.nn.sess, self.nn.pred_policy,
                      self.nn.inputs, training=False)
        nn_g = NNRecordedGame(b, p1, p2, 1)
        nn_g.initialize()
        nn_g.play_a_game(print_board=True)

    def update_training_data_(self, data, data_loop):
        data['state'] += data_loop['state']
        data['pi'] += data_loop['pi']
        data['z'] += data_loop['z']

    def prepare_data(self, raw_data):
        input_data = self.create_batches(np.array(raw_data['state']).reshape((-1, 6, 7, 3)))
        output_data_pi = self.create_batches(np.array(raw_data['pi']))
        output_data_z = self.create_batches(np.array(raw_data['z']).reshape((-1, 1)))
        return input_data, output_data_pi, output_data_z

    def create_batches(self, data):
        batches = []
        for j in range(0, len(data), CFG.batch_size):
            batches.append(data[j: j + CFG.batch_size])
        return batches
