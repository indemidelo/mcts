import numpy as np
from ..Game import Game, bcolors


class ConnectFour(Game):
    def __init__(self, rows=6, columns=7):
        super(ConnectFour, self).__init__(rows, columns)

    @staticmethod
    def input_shape():
        return [3, 6, 7]

    @staticmethod
    def policy_shape():
        return 7

    def __repr__(self):
        print()
        print('---------------------------------------')
        print(f'---------------{bcolors.HEADER}THE BOARD{bcolors.ENDC}---------------')
        print('---------------------------------------')
        for row, i in enumerate(self.board):
            print(' |', end=' ')
            for col, p in enumerate(i):
                value = f'{bcolors.OKBLUE}{self.history[row, col]:02d} {bcolors.ENDC}' \
                    if p == 1 else f'{bcolors.RED}{self.history[row, col]:02d} {bcolors.ENDC}' \
                    if p == -1 else f'{bcolors.GRAY}__ {bcolors.ENDC}'
                print(f'{value}|', end=' ')
            print()
        print('----1----2----3----4----5----6----7----')
        return ''

    def play_(self, player, index):
        row, col = self.index_to_pos(index)
        self.board[row, col] = float(player)
        self.plays += 1
        self.history[row, col] = self.plays
        if self.game_over(player, col, row):
            self.playing = False
            self.winner = player
            self.reward = 1
        if self.plays == self.board.shape[0] * self.board.shape[1]:
            self.playing = False

    def index_to_pos(self, index):
        column = self.board[:, index]
        row = max([j for j, v in enumerate(column) if v == 0])
        return row, index

    def game_over(self, *args):
        if self.horizontal_connect(*args):
            return True
        if self.vertical_connect(*args):
            return True
        if self.diagonal_connect(*args):
            return True
        return False

    def winning_combo(self, combo, player):
        # print('combo', combo, 'player', player)
        return all(j == player for j in combo)

    def horizontal_connect(self, player, col, pos):
        combo_start = max(0, col - 3)
        combo_end = min(6, col + 3)
        # print('Horizontal Combos')
        for j in range(combo_end - combo_start - 2):
            combo = self.board[pos, combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, player):
                self.playing = False
                return True
        return False

    def vertical_connect(self, player, col, pos):
        combo_start = max(0, pos - 3)
        combo_end = min(5, pos + 3)
        # print('Vertical Combos')
        for j in range(combo_end - combo_start - 2):
            combo = self.board[combo_start + j: combo_start + j + 4, col]
            if self.winning_combo(combo, player):
                self.playing = False
                return True
        return False

    def diagonal_connect(self, *args):
        if self.diagonal_connect_nw_to_se(*args):
            return True
        elif self.diagonal_connect_ne_to_sw(*args):
            return True
        return False

    def diagonal_connect_nw_to_se(self, player, col, pos):
        """
        North west to south east
        :param move:
        :return:
        """
        # combo_matrix = np.array([
        #     [0, 0, 0, 0, 0, 0, 0],
        #     [0, 1, 1, 1, 1, 1, 1],
        #     [0, 1, 2, 2, 2, 2, 2],
        #     [0, 1, 2, 3, 3, 3, 3],
        #     [0, 1, 2, 3, 4, 4, 4],
        #     [0, 1, 2, 3, 4, 5, 5]])
        diagonal = self.board.diagonal(col - pos)
        j_diag = min(col, pos)
        combo_start = max(0, j_diag - 3)
        combo_end = min(len(diagonal) - 1, j_diag + 3)
        # print('Diagonal NW combos')
        for j in range(combo_end - combo_start - 2):
            combo = diagonal[combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, player):
                self.playing = False
                return True
        return False

    def diagonal_connect_ne_to_sw(self, player, col, pos):
        """
        South west to north east
        :param move:
        :return:
        """
        specular_col = abs(6 - col)
        diagonal = np.flip(self.board, axis=-1).diagonal(specular_col - pos)
        j_diag = min(specular_col, pos)
        combo_start = max(0, j_diag - 3)
        combo_end = min(len(diagonal) - 1, j_diag + 3)
        # print('Diagonal NE combos')
        for j in range(combo_end - combo_start - 2):
            combo = diagonal[combo_start + j: combo_start + j + 4]
            if self.winning_combo(combo, player):
                self.playing = False
                return True
        return False

    def list_available_moves(self) -> list:
        av_moves = list()
        for j in range(self.board.shape[1]):
            if 0 in self.board[:, j]:
                av_moves.append(j)
        return av_moves
