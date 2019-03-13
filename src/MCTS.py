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
        self.N = 0
        self.tree = dict()

    def initialize(self):
        self.p1 = Player(1)
        self.p2 = Player(2)
        self.active_player = self.p1
        self.tree[self.board.hash] = State(
            None, self.active_player, self.board, 1)
        self.nn = NeuralNetwork()
        self.board.playing = True

    def move(self):
        for j in range(self.n_iter):
            leaf_hash, history = self.traverse_to_leaf()
            p, v = self.nn.eval(self.tree[leaf_hash])
            self.expand_leaf(leaf_hash, p)
            self.backpropagation(history, v)
            self.N += 1
        best_action = self.best_action()
        self.board.play_(self.active_player.name, best_action)
        self.log_something()
        self.active_player = self.opponent(self.active_player)

    def traverse_to_leaf(self):
        leaf_hash = self.board.hash
        history = []
        while self.tree[leaf_hash].sons:
            brothers = self.tree[leaf_hash].sons
            leaf = max(
                self.tree[leaf_hash].sons, key=lambda x: x.gain)
            leaf_hash = leaf.board.hash
            history.append({'hash': leaf_hash, 'brothers': brothers})
        return leaf_hash, history

    def expand_leaf(self, leaf_hash, p):
        leaf = self.tree[leaf_hash]
        active_player = self.active_player \
            if self.N == 0 else self.opponent(leaf.player)
        for action in leaf.board.list_available_moves():
            new_board = deepcopy(leaf.board)
            new_board.play_(active_player.name, action)
            new_state = State(action, active_player,
                              new_board, p[action])
            leaf.sons.append(new_state)
            # if new_state.board.hash not in self.tree:
            self.tree[new_board.hash] = new_state

    def backpropagation(self, history, v):
        for action in history:
            node = self.tree[action['hash']]
            n_all = sum([b.n for b in action['brothers']])
            node.update(v, n_all)

    def log_something(self):
        return

    def best_action(self):
        current_state = self.tree[self.board.hash]
        pi = [s.n ** (1 / self.tau) for s in current_state.sons]
        pi = [p / sum(pi) if sum(pi) else p for p in pi]
        actions = [s.action for s in current_state.sons]
        if sum(pi) == 0: pi = [1 / len(pi)] * len(pi)
        r = random.choices(actions, weights=pi, k=1)[0]
        return r

    def opponent(self, player):
        if player is None:
            return self.p1
        return self.p1 if player == self.p2 else self.p2
