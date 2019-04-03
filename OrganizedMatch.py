import random


class OrganizedMatch():
    def __init__(self, board, player_one, player_two):
        """
        A game between two players
        :param board:
        :param player_one:
        :param player_two:
        """
        self.board = board
        p1color = random.choice([1, -1])
        self.players = {p1color: player_one,
                        -p1color: player_two}

    def play_a_game(self, print_board=False):
        """ To play a game """
        player_color = 1
        if print_board:
            print(self.board)
        while self.board.playing and not self.board.full:
            player = self.players[player_color]
            player.play(self.board, player_color)
            if print_board:
                print(f'Player {player_color} move')
                print(self.board)
            player_color = -player_color


if __name__ == '__main__':
    from src.Board import Board
    from src.HumanPlayer import HumanPlayer
    from src.MCTS import SimulatedGame
    from src.config import CFG
    from src.NeuralNetwork import NeuralNetwork
    b = Board()
    ai = SimulatedGame(NeuralNetwork(), CFG.temp_thresh + 1)
    ai.nn.load_model('models/backup/funzionante/prova_190401_iter_30.ckpt')
    human = HumanPlayer('pippo')
    OrganizedMatch(b, ai, human).play_a_game(True)