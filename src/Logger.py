import random
from config import CFG


class Logger():
    def __init__(self):
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}
        self.pi_log = open('pi_log.txt', 'a')
        self.v_log = open('v_log.txt', 'a')

    def log_single_move(self, state, pi):
        self.saved_states['state'].append(state)
        self.saved_states['pi'].append(pi)

    def log_pi(self, pi, board_hash):
        self.pi_log.write(f'{pi};{board_hash}\n')

    def log_v(self, v, board_hash):
        self.v_log.write(f'{v};{board_hash}\n')

    def log_results(self, winner):
        for state in self.saved_states['state']:
            if winner is None:
                result = 0
            elif winner == state.player_color:
                result = 1
            else:
                result = -1
            self.saved_states['z'].append(result)

    def export_data_for_training(self, winner):
        """
        Export data for training only if the game is not a draw
        :param winner: 1 if White -1 if Black
        :return: (dict)
        """
        raw_data = {'state': list(), 'pi': list(), 'z': list()}
        if winner is None:
            return raw_data
        self.log_results(winner)
        n_states = len(self.saved_states['state'])
        indices = random.sample(
            range(n_states), int(n_states * CFG.train_split))
        for ind in indices:
            state = self.saved_states['state'][ind]
            player = state.player_color
            raw_data['state'].append(state.board.board_as_tensor(player))
            raw_data['pi'].append(list(self.saved_states['pi'][ind].values()))
            raw_data['z'].append(self.saved_states['z'][ind])
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}
        return raw_data
