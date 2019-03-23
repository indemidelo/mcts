class Player():
    def __init__(self, name, color=None):
        self.name = name
        self.color = color #1 for White, 2 for Black

    def __repr__(self):
        return str(self.name)
