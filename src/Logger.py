import random


class Logger():
    def __init__(self):
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}

    def log_single_game(self, state, pi):
        self.saved_states['state'].append(state)
        self.saved_states['pi'].append(pi)

    def log_results(self, board):
        for j, state in enumerate(self.saved_states['state']):
            result = 1 if board.winner == state.player else - 1
            self.saved_states['z'].append(result)

    def export_data_for_training(self, board, n_moves):
        self.log_results(board)
        n_states = len(self.saved_states['state'])
        indices = random.sample(
            range(n_states), min(n_states, n_moves))
        raw_data = {'input': list(), 'pi': list(), 'z': list()}
        for ind in indices:
            state = self.saved_states['state'][ind]
            player = state.player.name
            raw_data['input'].append(state.board.board_as_tensor(player))
            raw_data['pi'].append(list(self.saved_states['pi'][ind].values()))
            raw_data['z'].append(self.saved_states['z'][ind])
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}
        return raw_data
