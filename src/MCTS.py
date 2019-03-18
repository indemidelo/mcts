import random
import numpy as np
from copy import deepcopy
from src.Board import Board
from src.State import State
from src.Logger import Logger
from src.NeuralNetwork import NeuralNetwork


class SimulatedGame():
    def __init__(self, player_one, player_two,
                 n_iter, n_moves, c_puct=2, tau=1, eps=0.25):
        self.player_one = player_one
        self.player_two = player_two
        self.nn = NeuralNetwork()
        self.n_iter = n_iter
        self.logger = Logger()
        self.n_moves = n_moves
        self.c_puct = c_puct
        self.tau = tau
        self.eps = eps

    def initialize(self):
        self.N = 0
        self.board = Board()
        self.active_player, self.opponent = self.players_order()
        self.tree = State(None, self.active_player, self.board, 1)
        self.board.playing = True

    def players_order(self):
        return random.sample(
            (self.player_one, self.player_two), k=2)

    def play_a_game(self):
        self.initialize()
        while self.board.playing:
            self.move()
        training_data = self.logger.export_data_for_training(
            self.board, self.n_moves)
        return training_data

    def switch_players_(self):
        self.active_player, self.opponent = \
            self.opponent, self.active_player

    def move(self):
        # somma_mov = 0
        for j in range(self.n_iter):
            leaf, history = self.traverse_to_leaf()
            board_as_tensor = leaf.board.board_as_tensor(
                self.whos_opponent(leaf.player).name)
            p, v = self.nn.eval(board_as_tensor)
            if leaf.board.playing:
                self.expand_leaf(leaf, p)
            self.backpropagation(history, v)
            # somma_temp = sum([s.n for s in self.tree.children])
            # if self.N and somma_temp == somma_mov:
            #     print('qualcosa non va')
            self.N += 1
            # somma_mov = somma_temp
        # print('leaf board:', leaf.board)
        next_state, pi = self.play()
        self.logger.log_single_game(self.tree, pi)
        self.board.play_(self.active_player.name, next_state.action)
        self.switch_players_()
        self.tree = next_state

    def traverse_to_leaf(self):
        leaf = self.tree
        history = []
        while leaf.children:
            brothers = leaf.children
            leaf = max(leaf.children, key=lambda x: x.gain)
            history.append({'leaf': leaf, 'brothers': brothers})
        return leaf, history

    def expand_leaf(self, leaf, p):
        active_player = self.whos_opponent(leaf.player)
        for action in leaf.board.list_available_moves():
            new_board = deepcopy(leaf.board)
            new_board.play_(active_player.name, action)
            new_state = State(
                action, active_player, new_board, p[action])
            leaf.children.append(new_state)

    def backpropagation(self, history, v):
        for action in history:
            n_all = 1 + sum([b.n for b in action['brothers']])
            action['leaf'].update(v, n_all)

    def play(self):
        pi = {k: 0.0 for k in range(7)}  # todo change here to generalize over games
        # noise = iter(np.random.dirichlet([0.03] * 7, 1)[0])
        for s in self.tree.children:
            prob = s.n ** (1 / self.tau)
            # pi[s.action] = (1 - self.eps) * prob + self.eps * next(noise)
            pi[s.action] = prob
        pi = {k: v / sum(pi.values()) for k, v in pi.items()}
        action = random.choices(*zip(*pi.items()), k=1)[0]
        next_state = next((x for x in self.tree.children if x.action == action))
        return next_state, pi

    def whos_opponent(self, player):
        if self.N == 0:
            return self.active_player
        playerbase = [self.active_player, self.opponent]
        playerbase.remove(player)
        return playerbase[0]
