import random
import numpy as np
from copy import deepcopy
from src.config import CFG
from src.Board import Board
from src.State import State
from src.Logger import Logger


class SimulatedGame():
    def __init__(self, nn, temp_it=CFG.temp_thresh+1, player_name=None):
        self.nn = nn
        self.logger = Logger()
        self.temp_it = temp_it
        self.player_name = player_name
        self.tree = State(None, 1, Board(), p=1)

    def play_a_game(self, print_board=False):
        while self.tree.board.playing:
            self.move()
            if print_board:
                print(self.tree.board)
        training_data = self.logger.export_data_for_training(self.tree.board.winner)
        return training_data

    def move(self):
        self.search_()
        next_state, pi = self.sample_move()
        self.logger.log_single_move(self.tree, pi)
        self.tree = next_state
        self.tree.parent = None

    def play(self, board, player_color):
        """
        To play against an opponent
        :return:
        """
        self.tree = State(None, player_color, board, p=1)
        self.search_()
        next_state, _ = self.sample_move()
        board.play_(player_color, next_state.action)

    def search_(self):
        for j in range(CFG.num_mcts_sims):
            leaf = self.traverse_to_leaf()
            if leaf.board.playing:
                p, v = self.nn.eval(leaf)
                self.expand_leaf_(leaf, p)
            else:
                v = leaf.board.reward
            self.backpropagation_(leaf, v)

    def traverse_to_leaf(self):
        leaf = self.tree
        while leaf.children:
            leaf = max(leaf.children, key=lambda x: x.gain)
        return leaf

    def expand_leaf_(self, leaf, p):
        available_moves = leaf.board.list_available_moves()
        noise = iter(np.random.dirichlet(
            [CFG.dirichlet_alpha] * len(available_moves)))
        for action in available_moves:
            new_board = deepcopy(leaf.board)
            new_board.play_(leaf.player_color, action)
            prior = (1 - CFG.epsilon) * p[action] + CFG.epsilon * next(noise)
            new_state = State(action, -leaf.player_color, new_board, prior, leaf)
            leaf.children.append(new_state)

    def backpropagation_(self, leaf, v):
        while leaf.parent:
            n_all = 1 + sum([b.n for b in leaf.parent.children])
            leaf.update(v, n_all)
            leaf = leaf.parent
            v = -v

    def sample_move(self):
        pi = {k: 0.0 for k in range(7)}  # todo change here to generalize over games
        for s in self.tree.children:
            pi[s.action] = s.n ** (1 / CFG.temp_init)
        if self.temp_it < CFG.temp_thresh:
            pi = {k: v / sum(pi.values()) for k, v in pi.items()}
            action = random.choices(*zip(*pi.items()), k=1)[0]
        else:
            action = np.argmax(list(pi.values()))
            pi = {k: 0.0 if k != action else 1.0 for k in pi}
        next_state = next((x for x in self.tree.children if x.action == action))
        return next_state, pi

    def explode(self, root, depth):
        if root.children:
            for c in root.children:
                if c.n:
                    print(f'{"-" * depth}{c.n}')
                self.explode(c, depth + 1)
