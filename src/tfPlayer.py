import random
import numpy as np
from src.Move import Move
from src.Board import Board


class tfPlayer():
    def __init__(self, name: int, board: Board,
                 sess, pred_policy, input_placeholder,
                 training: bool = False):
        self.name = name
        self.board = board
        self.sess = sess
        self.pred_policy = pred_policy
        self.input_placeholder = input_placeholder
        self.training = training

    def play(self, fixed_move=None):
        if fixed_move is None:
            col = self.nn_move()
        else:
            col = fixed_move
        if col != -1:
            m = Move(self.name, self.board, col)
            m.play()
            win = self.board.winner == self.name
            return m, win
        else:
            self.board.playing = False
            self.board.full = True

    def nn_move(self):
        available_moves = self.board.list_available_moves()
        if not available_moves:
            return -1
        input_data = self.board.board_as_tensor(self.name)
        nn_policy = self.sess.run(
            [self.pred_policy], feed_dict={self.input_placeholder: input_data})[0][0]
        if not self.training:
            print(f'Player {self.name} policy: {nn_policy}')
        nn_policy_available = [v if j in available_moves else float('-inf')
                               for j, v in enumerate(nn_policy)]
        policy_dict = dict(zip(range(7), softmax(nn_policy_available)))
        if self.training:
            return random.choices(
                list(policy_dict.keys()), list(policy_dict.values()))[0]
        else:
            best_move = sorted(policy_dict.items(), key=lambda x: x[1])[-1][0]
            return best_move


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
