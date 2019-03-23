import random
import numpy as np
from copy import deepcopy
from src.Board import Board
from src.State import State
from src.Logger import Logger
from src.NeuralNetwork import NeuralNetwork


class SimulatedGame():
    def __init__(self, player_one, player_two,
                 n_iter, n_moves, c_puct=2, tau=1, eps=0.25,
                 dir_noise=0.03):
        self.player_one = player_one
        self.player_two = player_two
        self.nn = NeuralNetwork()
        self.n_iter = n_iter
        self.logger = Logger()
        self.n_moves = n_moves
        self.c_puct = c_puct
        self.tau = tau
        self.eps = eps
        self.dir_noise = dir_noise

    def initialize(self):
        self.N = 0
        self.active_player, self.opponent = self.choose_first_player()
        self.tree = State(None, self.active_player, Board(), 1)

    def choose_first_player(self):
        p1, p2 = random.sample((self.player_one, self.player_two), k=2)
        p1.color, p2.color = 1, 2
        return p1, p2

    def play_a_game(self):
        self.initialize()
        while self.tree.board.playing:
            self.move()
            print(self.tree.board)
        training_data = self.logger.export_data_for_training(
            self.tree.board.winner, self.n_moves)
        return training_data

    def switch_players_(self):
        self.active_player, self.opponent = \
            self.opponent, self.active_player

    def move(self):
        for j in range(self.n_iter):
            leaf, history = self.traverse_to_leaf()
            board_as_tensor = leaf.board.\
                board_as_tensor(leaf.player.color)
            p, v = self.nn.eval(board_as_tensor)
            if leaf.board.playing:
                self.expand_leaf_(leaf, p)
            self.backpropagation_(history, v)
            self.N += 1
        next_state, pi = self.sample_move()
        self.logger.log_single_move(self.tree, pi)
        self.switch_players_()
        self.tree = next_state

    def traverse_to_leaf(self):
        leaf = self.tree
        history = []
        while leaf.children:
            leaf = max(leaf.children, key=lambda x: x.gain)
            history.append({'leaf': leaf, 'brothers': leaf.children})
        return leaf, history

    def expand_leaf_(self, leaf, p):
        opponent = self.whos_opponent(leaf.player)
        available_moves = leaf.board.list_available_moves()
        noise = iter(np.random.dirichlet(
            [self.dir_noise] * len(available_moves)))
        for action in available_moves:
            new_board = deepcopy(leaf.board)
            new_board.play_(leaf.player.color, action)
            prior = (1 - self.eps) * p[action] + self.eps * next(noise)
            new_state = State(action, opponent, new_board, prior)
            leaf.children.append(new_state)

    def backpropagation_(self, history, v):
        for action in history:
            n_all = 1 + sum([b.n for b in action['brothers']])
            action['leaf'].update(v, n_all)

    def sample_move(self):
        pi = {k: 0.0 for k in range(7)}  # todo change here to generalize over games
        for s in self.tree.children:
            pi[s.action] = s.n ** (1 / self.tau)
        pi = {k: v / sum(pi.values()) for k, v in pi.items()}
        action = random.choices(*zip(*pi.items()), k=1)[0]
        next_state = next((x for x in self.tree.children if x.action == action))
        return next_state, pi

    def whos_opponent(self, player):
        if player == self.active_player:
            return self.opponent
        else:
            return self.active_player

    def explode(self, root, depth):
        if root.children:
            for c in root.children:
                if c.n:
                    print(f'{"-" * depth}{c.n}')
                self.explode(c, depth + 1)
