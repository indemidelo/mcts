from src.Move import Move


class HumanPlayer():
    def __init__(self, name):
        self.name = name

    def play(self, board, color):
        board.play_(color, self.human_move(board))

    def search_available_moves(self, board):
        av_moves = list()
        for j in range(board.board.shape[1]):
            if 0 in board.board[:, j]:
                av_moves.append(j)
        return av_moves

    def human_move(self, board):
        available_moves = self.search_available_moves(board)
        if not available_moves:
            return -1
        try:
            col = int(input('Your move:')) - 1
        except:
            print('Invalid move! Column not found')
            return self.human_move(board)
        if col in available_moves:
            return col
        print('Invalid move! The column is full')
        return self.human_move(board)
