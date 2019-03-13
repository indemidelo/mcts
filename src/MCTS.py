import random
from copy import deepcopy
from src.Board import Board
from src.State import State


class SimulatedGame():
    def __init__(self, player_one, player_two, neural_network,
                 n_iter, n_moves, logger, c_puct=2, tau=1):
        self.player_one = player_one
        self.player_two = player_two
        self.nn = neural_network
        self.n_iter = n_iter
        self.logger = logger
        self.n_moves = n_moves
        self.c_puct = c_puct
        self.tau = tau

    def initialize(self):
        self.N = 0
        self.board = Board()
        self.active_player, self.opponent = self.players_order()
        self.tree = {self.board.hash: State(
            None, self.active_player, self.board, 1)}
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
        for j in range(self.n_iter):
            leaf_hash, history = self.traverse_to_leaf()
            p, v = self.nn.eval(self.tree[leaf_hash])
            if self.tree[leaf_hash].board.playing:
                self.expand_leaf(leaf_hash, p)
            self.backpropagation(history, v)
            self.N += 1
        action, pi = self.play()
        self.logger.log_single_game(self.tree[self.board.hash], pi)
        self.board.play_(self.active_player.name, action)
        self.switch_players_()

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
        active_player = self.whos_opponent(leaf.player)
        for action in leaf.board.list_available_moves():
            new_board = deepcopy(leaf.board)
            new_board.play_(active_player.name, action)
            new_state = State(
                action, active_player, new_board, p[action])
            leaf.sons.append(new_state)
            self.tree[new_board.hash] = new_state

    def backpropagation(self, history, v):
        for action in history:
            state = self.tree[action['hash']]
            n_all = sum([b.n for b in action['brothers']])
            state.update(v, n_all)

    def play(self):
        current_state = self.tree[self.board.hash]
        pi = {k: 0.0 for k in range(7)}  # todo change here to generalize over games
        for s in current_state.sons:
            pi[s.action] = s.n ** (1 / self.tau)
        pi = {k: p / sum(pi) if sum(pi) else p for k, p in pi.items()}
        if sum(pi.values()) == 0.0:
            pi = {k: 1 / len(pi) for k in pi.keys()}
        move = random.choices(
            list(pi.keys()), list(pi.values()), k=1)[0]
        return move, pi

    def whos_opponent(self, player):
        if self.N == 0:
            return self.active_player
        playerbase = [self.active_player, self.opponent]
        playerbase.remove(player)
        return playerbase[0]
