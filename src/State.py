from math import sqrt
from config import CFG


class State():
    def __init__(self, action, player_color, board, p, parent=None):
        """
        Node of the Monte Carlo Tree
        :param action: last action performed by -player_color
        :param player_color: the next player to make a move
        :param board: the board before player_color's move
        :param p: prior probability
        :param parent: the parent state
        """
        self.action = action
        self.player_color = player_color
        self.board = board
        self.children = list()
        self.parent = parent
        self.p = p
        self.W = 0
        self.U = 0
        self.Q = float('inf')
        self.n = 0
        self.gain = self.Q + self.U

    def __str__(self):
        return str(f'p: {self.player_color} - '
                   f'a: {self.action} - n: {self.n} - '
                   f'gain: {self.gain}')

    def update(self, v, n_all):
        self.n += 1
        self.W += v
        self.U = CFG.c_puct * self.p * sqrt(n_all) / (self.n + 1)
        self.Q = self.W / self.n
        self.gain = self.Q + self.U
