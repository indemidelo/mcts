class Player():
    def __init__(self, name, color=None):
        self.name = name
        self.color = color #1 for Blue, -1 for Red

    def __repr__(self):
        return str(self.name)

    def play(self, game, color):
        game.play_(color, self.move(game))

    # to be implemented
    def move(self, game):
        pass

