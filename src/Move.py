class Move():
    def __init__(self, player, board, col):
        self.player = player
        self.board = board
        self.col = col
        self.row = None

    def play(self):
        self.row = self.board.play_(self.player, self.col)
