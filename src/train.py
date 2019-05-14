import numpy as np
import time
from datetime import datetime
from config import CFG
from src.MCTS import SimulatedGame
from src.Player import Player
from OrganizedMatch import OrganizedMatch

date = datetime.now().strftime("%y%m%d")
if CFG.framework == 'pytorch':
    from src.pytorch.NeuralNetwork import NeuralNetwork
elif CFG.framework == 'tensorflow':
    from src.tensorflow.NeuralNetwork import NeuralNetwork


class Training:
    def __init__(self, game, model_name=None):
        self.game = game
        self.net = NeuralNetwork(game)
        self.eval_net = NeuralNetwork(game)
        self.p1, self.p2 = Player(1), Player(2)
        self.init_train_data()
        if model_name:
            self.net.load_model(model_name)

    def init_train_data(self):
        self.train_data = {'state': [], 'pi': [], 'z': []}

    def train(self):

        for i in range(CFG.num_iterations):

            mean_el = 0
            for g in range(CFG.num_games):
                start_time = time.time()
                simgame = SimulatedGame(self.net, g + i * CFG.num_iterations)
                training_data_loop = simgame.play_a_game()
                self.update_training_data_(training_data_loop)
                elapsed = time.time() - start_time
                mean_el += elapsed / CFG.num_games
                print(f'Game {g + 1} in iter {i + 1} '
                      f'won by player {simgame.tree.board.winner}'
                      f' - elapsed: {round(elapsed, 2)}s')
            print(f'Mean elapsed time for one iteration = {round(mean_el, 2)}')

            print('Data used for training: ', len(self.train_data['state']))
            self.net.train(*self.prepare_data())
            # self.test()

            self.init_train_data()

            # elif (i + 1) % CFG.checkpoint == 0:
            #     filename = f'{CFG.model_directory}prova_{date}_iter_{i + 1}.ckpt'
            #     self.eval_networks_()
            #     self.net.save_model(filename)
            #     self.net.load_model(filename)

    def eval_networks_(self):
        print('Networks evaluation')
        ai_new = SimulatedGame(self.net, player_name='new')
        ai_old = SimulatedGame(self.eval_net, player_name='old')
        wins = 0
        num_eval_games = CFG.num_eval_games
        for j in range(CFG.num_eval_games):
            match = OrganizedMatch(self.game, ai_new, ai_old)
            winner = match.play_a_game()
            print(f'Match {j + 1} won by {winner}')
            if winner == 'new':
                wins += 1
            elif winner is None:
                num_eval_games -= 1
        if num_eval_games and wins > CFG.eval_win_rate * num_eval_games:
            self.net.age += 1
            print(f'Stronger network trained :) WR='
                  f'{round(wins / num_eval_games, 2)}'
                  f' network age={self.net.age}')
            ai_new.logger.log_v('stronger', 'network trained')
            ai_new.logger.log_pi('stronger', 'network trained')
            self.net.save_model(f'{CFG.model_directory}old/old_nn')
            self.eval_net.load_model(f'{CFG.model_directory}old/old_nn')
            self.init_train_data()
        elif num_eval_games:
            print(f'Weaker network trained :( WR='
                  f'{round(wins / num_eval_games, 2)}'
                  f' network age={self.net.age}')
            self.net.load_model(f'{CFG.model_directory}old/old_nn')
            if CFG.data_waste:
                self.init_train_data()
        else:
            print('All draws! Force update :/'
                  f' network age={self.net.age}')
            self.net.save_model(f'{CFG.model_directory}old/old_nn')
            self.eval_net.load_model(f'{CFG.model_directory}old/old_nn')
            self.init_train_data()

    def test(self, model_filename=None):
        if model_filename:
            self.net.load_model(model_filename)
        SimulatedGame(self.net).play_a_game(print_board=True)

    def update_training_data_(self, data_loop):
        self.train_data['state'] += data_loop['state']
        self.train_data['pi'] += data_loop['pi']
        self.train_data['z'] += data_loop['z']

    def prepare_data(self):
        input_data = self.create_batches(
            np.array(self.train_data['state']).reshape([-1] + self.game.input_shape()))
        output_data_pi = self.create_batches(np.array(self.train_data['pi']))
        output_data_z = self.create_batches(np.array(self.train_data['z']))
        return input_data, output_data_pi, output_data_z

    def create_batches(self, data):
        batches = []
        for j in range(0, len(data), CFG.batch_size):
            batches.append(data[j: j + CFG.batch_size])
        return batches
