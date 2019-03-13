import random
import time
import numpy as np
import tensorflow as tf
from src.tfPlayer import tfPlayer
from src.NNGame import NNRecordedGame
from src.NNGame_mp import NNRecordedGame_mp
from src.Board import Board
from src.tensorflow_network import AlphaGo19Net


def sample_player_moves(nn_game, player, batch_size):
    input_data, output_data = [], []
    games_played = len(nn_game.history[player.name]['moves'])
    games_to_sample = min(games_played, int(batch_size / 2))
    sampled_indices = random.sample(range(games_to_sample), int(batch_size / 2))
    for j in sampled_indices:
        input_data.append((nn_game.history[player.name]['states'][j]))
        output_data.append(nn_game.history[player.name]['rollout_pol'][j])
    input_data = np.asarray(input_data).reshape((4, 6, 7, 2))
    output_data = np.asarray(output_data).reshape((4, 7))
    return input_data, output_data


def get_all_player_moves(nn_game, player):
    input_data = np.asarray(nn_game.history[player.name]['states'])
    input_data = input_data.reshape((-1, 6, 7, 2))
    output_data = np.array(nn_game.history[player.name]['rollout_pol'])
    output_data = output_data.reshape((-1, 7))
    return input_data, output_data


def train(n_res_blocks: int, num_epochs: int, num_games: int,
          batch_size: int, learning_rate: float, mcts_iter: int):

    # Placeholder for input_data
    inputs = tf.placeholder(tf.float32, [None, 6, 7, 2], name='InputData')

    # Placeholder for p
    p = tf.placeholder(tf.float32, [None, 7], name='p')

    # Neural Network
    pred, loss, optimizer, me = AlphaGo19Net(
        inputs, p, n_res_blocks, learning_rate)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("Loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("Mean Error", me)

    # Initialize the variables
    init = tf.global_variables_initializer()

    print("Run the command line:\n"
          "--> tensorboard --logdir=/tmp/tensorflow_logs \n"
          "Then open http://0.0.0.0:6006/ into your web browser")

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    # Train the model
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        start_time = time.time()

        for e in range(num_games):

            # Create the board
            b = Board()

            # Create the players and the game
            p1 = tfPlayer(1, b, sess, pred, inputs, training=True)
            p2 = tfPlayer(2, b, sess, pred, inputs, training=True)
            # nn_g = NNRecordedGame(b, p1, p2, mcts_iter)
            nn_g = NNRecordedGame(b, p1, p2, mcts_iter)
            nn_g.initialize()

            # Play the game
            nn_g.play_a_game(print_board=False)

            # Collect the results
            input_p1, output_p1 = get_all_player_moves(nn_g, p1)
            input_p2, output_p2 = get_all_player_moves(nn_g, p2)
            input_data = np.concatenate([input_p1, input_p2])
            output_data = np.concatenate([output_p1, output_p2])

            # Training cycle
            for epoch in range(num_epochs):
                # feed dict
                feed_dict = {inputs: input_data, p: output_data}

                # fit the model
                _, c, mean_error = sess.run(
                    [optimizer, loss, me],
                    feed_dict=feed_dict)


                # Save the model
                # saver.save(
                #     sess, f"models/my_little_model_game_{e}_epoch_{epoch}.ckpt")

                # if (epoch + 1) % 25 == 0:
                print(f"Epoch: {epoch+1} - cost= {c}")
                print('Mean Error: ', mean_error)

            end_time = time.time()
            print(f"{end_time - start_time} - Game: {e+1} completed")
            start_time = end_time
    return sess, pred, inputs
