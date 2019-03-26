import random
import numpy as np
from copy import deepcopy
from src.config import CFG
from src.Board import Board
from src.State import State
from src.Logger import Logger
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


class SimulatedGame():
    def __init__(self):
        self.nn = NeuralNetwork()
        self.logger = Logger()
        self.player_one = Player(1)
        self.player_two = Player(2)

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
        training_data = self.logger.export_data_for_training(self.tree.board.winner)
        return training_data

    def switch_players_(self):
        self.active_player, self.opponent = \
            self.opponent, self.active_player

    def move(self):
        self.search_()
        next_state, pi = self.sample_move()
        self.logger.log_single_move(self.tree, pi)
        self.switch_players_()
        self.tree = next_state
        self.tree.parent = None

    def search_(self):
        for j in range(CFG.num_mcts_sims):
            leaf = self.traverse_to_leaf()
            if leaf.board.playing:
                p, v = self.nn.eval(leaf)
                self.expand_leaf_(leaf, p)
            else:
                v = -leaf.board.reward
            self.backpropagation_(leaf, v)
            self.N += 1

    def traverse_to_leaf(self):
        leaf = self.tree
        while leaf.children:
            leaf = max(leaf.children, key=lambda x: x.gain)
        return leaf

    def expand_leaf_(self, leaf, p):
        opponent = self.whos_opponent(leaf.player)
        available_moves = leaf.board.list_available_moves()
        noise = iter(np.random.dirichlet(
            [CFG.dirichlet_alpha] * len(available_moves)))
        for action in available_moves:
            new_board = deepcopy(leaf.board)
            new_board.play_(leaf.player.color, action)
            prior = (1 - CFG.epsilon) * p[action] + CFG.epsilon * next(noise)
            new_state = State(action, opponent, new_board, prior, leaf)
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
