from config import CFG
from copy import deepcopy
import random


def play_a_game(board, player_color, move):
    """ To play a game """
    board.play_(player_color, move)
    if board.winner:
        return board.winner, board.remaining_moves()
    else:
        return semi_random_game(board, -player_color)


def semi_random_game(board, player_color):
    while board.playing:
        lag = board.list_available_moves()

        # Check if there are winning moves
        for check in lag:
            board_clone = deepcopy(board)
            board.play_(player_color, check)
            if board_clone.winner:
                return board_clone.winner, board_clone.remaining_moves()

        # Else move random
        random_move = random.choice(lag)
        board.play_(player_color, random_move)
        if board.winner:
            return board.winner, board.remaining_moves()

        player_color = -player_color

    return 0, 0  # Game ended in a draw


def random_game(board, player_color):
    while board.playing:
        random_move = random.choice(board.list_available_moves())
        board.play_(player_color, random_move)
        if board.winner:
            return board.winner, board.remaining_moves()
        player_color = -player_color
    return 0, 0  # Game ended in a draw


def compute_easy_v(scores):
    # v is the total scores of the most winning player
    # divided by the absolute sum of all scores
    pass


def compute_v(scores):
    # v is the total scores of the most winning player
    # divided by the absolute sum of all scores
    positive_scores = sum(map(lambda x: x if x > 0 else 0, scores))
    negative_scores = sum(map(lambda x: abs(x) if x < 0 else 0, scores))
    if positive_scores > negative_scores:
        return positive_scores / (positive_scores + negative_scores)
    else:
        return -negative_scores / (positive_scores + negative_scores)


class PureRLAgent(object):
    def __init__(self, game):
        self.game = game
        self.age = 0

    def eval(self, state):
        # Fire n games for each available move
        # Rollback policy evaluation of state `state`
        available_moves = state.board.list_available_moves()
        scores = [0 for _ in range(self.game.policy_shape())]
        for move in available_moves:

            for j in range(CFG.num_mcts_sims):
                new_board = deepcopy(state.board)
                winner, remaining_moves = play_a_game(new_board, state.player_color, move)

                # the winner is the playing player (opposite of next player's color)
                if winner == -state.player_color:
                    scores[move] += remaining_moves
                # the winner is the opponent (player_color is the player doing the next move)
                elif winner == state.player_color:
                    scores[move] -= remaining_moves

        # pos_scores = deepcopy(scores)
        # for j in scores:
        #     if j < 0:
        #         pos_scores = [i + abs(j) for i in pos_scores]
        # p = [j / sum(pos_scores) if sum(pos_scores) else j for j in pos_scores]
        # v = compute_v(scores) if sum(scores) else 0

        pos_scores = deepcopy(scores)
        if min(pos_scores) < 0:
            pos_scores = [j + abs(min(pos_scores)) for j in pos_scores]
        p = [j / sum(pos_scores) if sum(pos_scores) else j for j in pos_scores]
        v = compute_v(scores) if sum(scores) else 0
        return p, v
