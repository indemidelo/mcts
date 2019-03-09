class State():
    def __init__(self, player, board, father, p):
        self.player = player
        self.board = board
        self.father = father
        self.sons = list()
        self.p = p
        self.w = 0
        self.v = 0
        self.q = 0
        self.n = 0

    def set_sons(self, sons):
        self.sons = sons