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


class Board():
    def __init__(self, rows=6, columns=7):
        self.board = np.zeros((rows, columns), dtype=float)
        self.history = np.zeros((rows, columns), dtype=int)
        self.playing = True
        self.plays = 0
        self.full = False
        self.winner = None
        self.reward = 0

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

    @property
    def hash(self):
        return ''.join(''.join(str(int(k)) for k in j) for j in self.board)

    def play_(self, player, col):
        if self.board[:, col][0] != 0:
            print(f'The {col + 1} column is already full')
            return -1
        else:
            pos = self.find_free_spot(col)
            self.board[pos, col] = float(player)
            self.plays += 1
            self.history[pos, col] = self.plays
            if self.plays == self.board.shape[0] * self.board.shape[1]:
                self.full = True
                self.playing = False
            else:
                if self.check_connect(player, col, pos):
                    self.playing = False
                    self.winner = player
                    self.reward = 1
            return pos

    def find_free_spot(self, col):
        column = self.board[:, col]
        return max([j for j, v in enumerate(column) if v == 0])

    def check_connect(self, *args):
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

    def uniform_board(self):
        new_board = []
        for j in self.board:
            row = []
            for i in j:
                row.append(int(1 / (0.5 * i) if i else i))
            new_board.append(row)
        return new_board

    def list_available_moves(self) -> list:
        av_moves = list()
        for j in range(self.board.shape[1]):
            if 0 in self.board[:, j]:
                av_moves.append(j)
        return av_moves

    def input_data_board(self):
        p1board, p2board = list(), list()
        for j in self.board:
            row1, row2 = list(), list()
            for i in j:
                row1.append(float(i == 1))
                row2.append(float(i == 2))
            p1board.append(row1)
            p2board.append(row2)
        return p1board, p2board

    def board_as_tensor(self, player):
        p1board, p2board = self.input_data_board()
        player_matrix = np.zeros((6, 7))
        if player == 1:
            player_matrix += 1
            game_matrix = np.array(
                (p1board, p2board, player_matrix))
        else:
            game_matrix = np.array(
                (p2board, p1board, player_matrix))
        return game_matrix.reshape((1, 6, 7, 3))
