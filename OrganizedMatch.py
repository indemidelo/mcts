import random


class OrganizedMatch():
    def __init__(self, game, player_one, player_two, verbose=False):
        """
        A game between two players
        :param game:
        :param player_one:
        :param player_two:
        """
        self.game = game
        p1color = random.choice([1, -1])
        self.players = {p1color: player_one,
                        -p1color: player_two}
        if verbose:
            print(f'{self.players[1].player_name} plays as Blue\n'
                  f'{self.players[-1].player_name} plays as Red\n'
                  f'Blue starts')

    def play_a_game(self, print_board=False):
        """ To play a game """
        player_color = 1
        g = self.game()
        if print_board:
            print(g)
        while g.playing:
            player = self.players[player_color]
            player.play(g, player_color)
            if print_board:
                print(g)
            player_color = -player_color
        if g.winner:
            winner = self.players[g.winner].player_name
            print(f'{winner} won')
            return winner
        return None


if __name__ == '__main__':
    from games.tic_tac_toe.TicTacToe import TicTacToe
    from src.HumanPlayer import HumanPlayer
    from src.MCTS import SimulatedGame
    from src.NeuralNetwork import NeuralNetwork
    # ai = SimulatedGame(NeuralNetwork(TicTacToe), player_name='ai')
    # ai.nn.load_model('../models/prova_iter_50.ckpt')
    human = HumanPlayer('Castor')
    human2 = HumanPlayer('Pollux')
    OrganizedMatch(TicTacToe, human2, human).play_a_game(True)

