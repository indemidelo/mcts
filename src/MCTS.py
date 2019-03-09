from src.Board import Board
from src.State import State
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


class MCTS():
    def __init__(self, n_iter):
        self.b = Board()
        self.n_iter = n_iter
        self.N = 0
        self.nodes_to_eval = list()
        self.tree = dict()

    def initialize(self):
        self.p1 = Player(1)
        self.p2 = Player(2)
        self.active_player = self.p1
        s0 = self.b.hash
        self.tree[s0] = State(
            self.active_player, self.b, None, 1)
        self.nn = NeuralNetwork()
        sons = list()
        p, v = self.nn.eval(s0)
        for action in self.b.available_moves():
            sons.append(State(self.active_player,
                              self.b.play(self.active_player, action),
                              s0, p[action]))
        self.tree[s0].set_sons(sons)
        self.tree[s0].v = v
        self.tree[s0].n += 1
        self.N += 1

    def loop(self):
        for j in range(self.n_iter):
            leaf = self.traverse_to_leaf()
            p, v = self.nn.eval(leaf)
            self.expanse_leaf(leaf, p, v)
            self.backpropagation(leaf, p, v)
            self.log_something()
        return self.best_action()

    def traverse_to_leaf(self):
        return None

    def expanse_leaf(self, leaf, p, v):
        return

    def backpropagation(self, leaf, p, v):
        return

    def log_something(self):
        return

    def best_action(self):
        return
