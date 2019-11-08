from src.Player import Player


class HumanPlayer(Player):

    def move(self, game):
        available_moves = game.list_available_moves()
        if not available_moves:
            return -1
        try:
            col = int(input(f'{self.name} move:')) - 1
        except:
            print('Invalid move! Column not found')
            return self.move(game)
        if col in available_moves:
            return col
        print('Invalid move! The column is full')
        return self.move(game)
