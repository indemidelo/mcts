class Board():
    def __init__(self):
        self.hash = 'hash'

    def available_moves(self):
        return (0, 1, 2, 3)

    def play(self, player, action):
        return 1
