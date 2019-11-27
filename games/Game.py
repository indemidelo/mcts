from config import CFG
import numpy as np


class bcolors:
    CYANO = '\033[96m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


class Game(object):
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((rows, columns), dtype=float)
        self.history = np.zeros((rows, columns), dtype=int)
        self.playing = True
        self.plays = 0
        self.winner = None
        self.reward = 0

    @staticmethod
    def n_channels():
        return CFG.n_channels_per_board

    @staticmethod
    def input_shape():
        return

    @staticmethod
    def policy_shape():
        return

    def __repr__(self):
        pass

    @property
    def hash(self):
        return ''.join(''.join(str(int(k)) for k in j) for j in self.board)

    def play_(self, player, index):
        return

    def index_to_pos(self, index):
        return

    def game_over(self, *args):
        return

    def remaining_moves(self):
        return len([j for i in self.board for j in i if j == 0])

    def input_data_board(self):
        p1board, p2board = list(), list()
        for j in self.board:
            row1, row2 = list(), list()
            for i in j:
                row1.append(float(i == 1))
                row2.append(float(i == -1))
            p1board.append(row1)
            p2board.append(row2)
        return p1board, p2board

    def board_repr(self, player):
        """
        Format the board in a customizable layout.
            CFG.n_channels_per_board = 1 : Player 1 always vs Player -1
            CFG.n_channels_per_board = 3 : Board layout and a layer for the
                                           current player
            CFG.n_channels_per_board = 3 : AlphaZero layout

        :param player: (int) player id. Can be 1 or -1
        :return: np.array with board representation
        """
        if CFG.n_channels_per_board == 1:
            return self.essential_board_repr(player)
        elif CFG.n_channels_per_board == 2:
            return self.simpler_board_repr(player)
        else:
            return self.board_repr_alphazero(player)

    def essential_board_repr(self, player):
        """
        One single filter. If player one has to move return the board,
        otherwise invert the color of all the pieces on the board
        :param player: (int) player id
        :return: np.array with board representation
        """
        if player == 1:
            return np.array([self.board])
        else:
            return np.array([-self.board])

    def board_repr_alphazero(self, player):
        """
        Format the board in the AlphaZero layout:
        1st layer = Player one
        2nd layer = Player two
        3rd layer = matrix of ones if P1 is the active player,
                    zeroes if P2 is the active player
        :param player: id of the active player
        :return: arrays with board representation
        """
        p1board, p2board = self.input_data_board()
        player_matrix = np.zeros((self.rows, self.columns))
        if player == 1:
            player_matrix += 1
        game_matrix = np.array(
            (p1board, p2board, player_matrix))
        return np.array([game_matrix])

    def simpler_board_repr(self, player=1):
        """
        Format a vector of boards in the fully connected layout
        :param player:
        :return:
        """
        player_matrix = np.zeros((self.rows, self.columns))
        if player == 1:
            player_matrix += 1
        return np.array((self.board, player_matrix))
