class HumanPlayer():
    def __init__(self, name):
        self.player_name = name

    def play(self, game, color):
        game.play_(color, self.human_move(game))

    def human_move(self, game):
        available_moves = game.list_available_moves()
        if not available_moves:
            return -1
        try:
            col = int(input(f'{self.player_name} move:')) - 1
        except:
            print('Invalid move! Column not found')
            return self.human_move(game)
        if col in available_moves:
            return col
        print('Invalid move! The column is full')
        return self.human_move(game)
