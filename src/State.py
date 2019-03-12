from math import sqrt


class State():
    def __init__(self, action, player, board, father_hash, p, c_puct=2):
        self.action = action
        self.player = player
        self.board = board
        self.father_hash = father_hash
        self.sons = list()
        self.c_puct = c_puct
        self.p = p
        self.W = 0
        self.U = 0
        self.Q = 0
        self.n = 0
        self.gain = self.U + self.Q

    def __iter__(self):
        yield self.gain

    def update(self, v, n_others):
        self.n += 1
        self.W += v
        self.U = self.c_puct * self.p * sqrt(n_others) / (self.n + 1)
        self.Q = self.W / self.n
