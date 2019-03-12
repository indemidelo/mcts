import random
from copy import deepcopy
from src.Board import Board
from src.State import State
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


class MCTS():
    def __init__(self, n_iter, c_puct=2, tau=1):
        self.n_iter = n_iter
        self.c_puct = c_puct
        self.tau = tau
        self.board = Board()
        self.current_state_hash = self.board.hash
        self.N = 0
        self.nodes_to_eval = list()
        self.tree = dict()

    def initialize(self):
        self.p1 = Player(1)
        self.p2 = Player(2)
        self.active_player = self.p1
        self.tree[self.current_state_hash] = State(
            None, self.active_player, self.board, None, 1)
        self.nn = NeuralNetwork()
        self.board.playing = True

    def loop(self):
        for j in range(self.n_iter):
            leaf_hash = self.traverse_to_leaf()
            p, v = self.nn.eval(self.tree[leaf_hash])
            self.expanse_leaf(leaf_hash, p)
            self.backpropagation(leaf_hash, v)
            self.N += 1
        best_action = self.best_action()
        self.board.play_(self.active_player.name, best_action)
        self.current_state_hash = self.board.hash
        self.log_something()
        self.active_player = self.p1 \
            if self.active_player == self.p2 else self.p2

    def traverse_to_leaf(self):
        leaf_hash = self.current_state_hash
        while self.tree[leaf_hash].sons:
            leaf = max(
                self.tree[leaf_hash].sons, key=lambda x: x.gain)
            leaf_hash = leaf.board.hash
        return leaf_hash

    def expanse_leaf(self, leaf_hash, p):
        leaf = self.tree[leaf_hash]
        for action in leaf.board.list_available_moves():
            new_board = deepcopy(leaf.board)
            new_board.play_(self.active_player.name, action)
            new_state = State(action, self.active_player,
                              new_board, leaf_hash, p[action])
            leaf.sons.append(new_state)
            self.tree[new_state.board.hash] = new_state

    def backpropagation(self, leaf_hash, v):
        while leaf_hash != self.current_state_hash:
            leaf = self.tree[leaf_hash]
            father_hash = self.tree[leaf_hash].father_hash
            brothers = self.tree[father_hash].sons
            n_others = sum([b.n for b in brothers]) - leaf.n
            leaf.update(v, n_others)
            leaf_hash = father_hash

    def log_something(self):
        return

    def best_action(self):
        current_state = self.tree[self.current_state_hash]
        pi = [s.n ** (1 / self.tau) for s in current_state.sons]
        pi = [p / sum(pi) for p in pi]
        actions = [s.action for s in current_state.sons]
        r = random.choices(actions, pi)[0]
        print(r)
        return r
