import numpy as np
from copy import deepcopy
from src.Board import Board
from src.config import CFG
from src.MCTS import SimulatedGame
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork
from src.OrganizedMatch import OrganizedMatch


class Training():
    def __init__(self):
        self.nn = NeuralNetwork()
        self.prev_nn = NeuralNetwork()
        self.p1, self.p2 = Player(1), Player(2)

    def train(self, model_filename=None):

        for i in range(CFG.num_iterations):

            training_data = {'state': [], 'pi': [], 'z': []}

            for g in range(CFG.num_games):
                simgame = SimulatedGame(self.nn, g + i * CFG.num_iterations)
                training_data_loop = simgame.play_a_game()
                self.update_training_data_(training_data, training_data_loop)
                print(f'Game {g + 1} in iter {i + 1} won by player {simgame.tree.board.winner}')

            self.nn.train(*self.prepare_data(training_data))
            self.test()

            if (i + 1) % CFG.checkpoint == 0:
                filename = f'{CFG.model_directory}prova_190401_iter_{i + 1}.ckpt'
                self.eval_networks_(filename)
                self.nn.save_model(filename)
                self.nn.load_model(filename)

        if model_filename:
            self.nn.save_model(model_filename)

    def eval_networks_(self, filename):
        print('Networks evaluation')
        ai_new = SimulatedGame(self.nn, 'new')
        ai_old = SimulatedGame(self.prev_nn, 'old')
        winrate = 0
        for _ in range(CFG.num_eval_games):
            b = Board()
            match = OrganizedMatch(b, ai_new, ai_old)
            if match.play_a_game().player_name == 'new':
                winrate += 1 / CFG.num_eval_games
                if winrate <= CFG.eval_win_rate:
                    print(f'Stronger network trained :) WR={round(winrate, 2)}')
                    self.nn = self.prev_nn
                else:
                    print(f'Weaker network trained :( WR={round(winrate, 2)}')
                    self.prev_nn = self.nn

    def test(self, model_filename=None):
        if model_filename:
            self.nn.load_model(model_filename)
        SimulatedGame(self.nn).play_a_game(print_board=True)

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
