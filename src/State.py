from math import sqrt


class State():
    def __init__(self, action, player, board, p, c_puct=2):
        self.action = action
        self.player = player
        self.board = board
        self.sons = list()
        self.c_puct = c_puct
        self.p = p
        self.W = 0
        self.U = 0
        self.Q = float('inf')
        self.n = 0

    @property
    def gain(self):
        return self.Q + self.U

    def update(self, v, n_all):
        self.n += 1
        self.W += v
        self.U = self.c_puct * self.p * sqrt(n_all) / (self.n + 1)
        self.Q = self.W / self.n
