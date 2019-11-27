import numpy as np
from ..Game import Game, bcolors


class TicTacToe(Game):
    def __init__(self, rows=3, columns=3):
        super(TicTacToe, self).__init__(rows, columns)

    @staticmethod
    def input_shape():
        return [Game.n_channels(), 3, 3]

    @staticmethod
    def policy_shape():
        return 9

    def __repr__(self):
        print()
        print('------------------')
        print(f'-----{bcolors.HEADER}THE BOARD{bcolors.ENDC}----')
        print('------------------')
        for row, i in enumerate(self.board):
            print(' |', end=' ')
            for col, p in enumerate(i):
                value = f'{bcolors.OKBLUE}{self.history[row, col]:02d} {bcolors.ENDC}' \
                    if p == 1 else f'{bcolors.RED}{self.history[row, col]:02d} {bcolors.ENDC}' \
                    if p == -1 else f'{bcolors.GRAY}{row * 3 + col + 1:02d} {bcolors.ENDC}'
                print(f'{value}|', end=' ')
            print()
        print('---1----2----3----')
        return ''

    def play_(self, player, index) -> None:
        row, col = self.index_to_pos(index)
        self.board[row, col] = float(player)
        self.plays += 1
        self.history[row, col] = self.plays
        if self.game_over(player, row, col):
            self.playing = False
            self.winner = player
            self.reward = 1
        if self.plays == self.rows * self.columns:
            self.playing = False

    def index_to_pos(self, index) -> tuple:
        row = int(index / 3)
        col = int(index % 3)
        return row, col

    def game_over(self, *args) -> bool:
        if self.horizontal_connect(*args):
            return True
        if self.vertical_connect(*args):
            return True
        if self.diagonal_connect(*args):
            return True
        return False

    def winning_combo(self, combo, player) -> bool:
        if all(j == player for j in combo):
            self.playing = False
            return True
        return False

    def horizontal_connect(self, player, row, col) -> bool:
        combo = self.board[row, :]
        return self.winning_combo(combo, player)

    def vertical_connect(self, player, row, col):
        combo = self.board[:, col]
        return self.winning_combo(combo, player)

    def diagonal_connect(self, player, row, col) -> bool:
        win = False
        if row == col:
            combo = self.board.diagonal()
            win = self.winning_combo(combo, player)
        if not win and row + col == 2:
            combo = np.rot90(self.board).diagonal()
            win = self.winning_combo(combo, player)
        return win

    def list_available_moves(self) -> list:
        return [j for j, v in enumerate(self.board.reshape(-1)) if v == 0]
