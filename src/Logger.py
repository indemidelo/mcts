from src.singleton import Singleton
import random


class Logger(metaclass=Singleton):
    def __init__(self):
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}

    def log_single_move(self, state, pi):
        self.saved_states['state'].append(state)
        self.saved_states['pi'].append(pi)
        print(f'pi: {pi}')
        print(f'Active player: {state.player.name} '
              f'- action: {state.action}')
        # print(f'Board:')
        # print(f'{state.board}')

    def log_results(self, winner):
        for state in self.saved_states['state']:
            if winner is None:
                result = 0
            elif winner == state.player.name:
                result = 1
            else:
                result = -1
            self.saved_states['z'].append(result)

    def export_data_for_training(self, winner, n_moves):
        self.log_results(winner)
        n_states = len(self.saved_states['state'])
        indices = random.sample(
            range(n_states), min(n_states, n_moves))
        raw_data = {'input': list(), 'pi': list(), 'z': list()}
        for ind in indices:
            state = self.saved_states['state'][ind]
            player = state.player.name
            raw_data['input'].append(state.board.board_as_tensor(player))
            raw_data['pi'].append(list(self.saved_states['pi'][ind].values()))
            print(raw_data['pi'][-1])
            raw_data['z'].append(self.saved_states['z'][ind])
        self.saved_states = {'state': list(), 'pi': list(), 'z': list()}
        return raw_data
