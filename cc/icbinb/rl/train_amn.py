# Code adapted for CC-AMN from:
# https://github.com/EvolvedSquid/tutorials/tree/master/dqn

from config import (BATCH_SIZE, CLIP_REWARD,
                    EVAL_LENGTH, FRAMES_BETWEEN_EVAL, INPUT_SHAPE,
                    LEARNING_RATE,
                    MAX_EPISODE_LENGTH, MAX_NOOP_STEPS,
                    PRIORITY_SCALE,
                    TENSORBOARD_DIR, TOTAL_FRAMES, UPDATE_FREQ, USE_PER,
                    WRITE_TENSORBOARD, SEED)

from train_dqn import GameWrapper, Agent, build_q_network, ReplayBuffer

import numpy as np

import random
import os
import json
import time

import tensorflow as tf

from datetime import datetime

import sys
sys.path.append('../cc')
from cc_funcs import cc_log_prob, cc_mean

MIN_REPLAY_BUFFER_SIZE = 5000
MEM_SIZE = 100000


if SEED is not None:
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

# Available only in tf>=2.2.0
# EAGER = False # Change to True to run eagerly for debugging
# if EAGER:
#     tf.config.run_functions_eagerly(True)


def find_saved_experts(env_names):
    paths = []
    for env_name in env_names:
        path = './saves/' + env_name + '/'
        saves = os.listdir(path)
        saves.sort()
        path = path + saves[-1]
        paths.append(path)

    return paths

def get_actions(game_wrappers):
    res = []
    for game_wrapper in game_wrappers:
        actions = game_wrapper.env.unwrapped.get_action_meanings()
        res.extend(actions)

    return list(dict.fromkeys(res))

class AMN(object):
    """Implements the AMN agent"""
    def __init__(self,
                 amn,
                 replay_buffers,
                 n_actions,
                 input_shape=(84, 84),
                 temperature=1.0,
                 loss='XE',
                 batch_size=32,
                 history_length=4,
                 eps_initial=1,
                 eps_final=0.1,
                 eps_final_frame=0.01,
                 eps_evaluation=0.0,
                 eps_annealing_frames=1000000,
                 replay_buffer_start_size=50000,
                 max_frames=25000000,
                 use_per=True):
        """
        Arguments:
            dqn: A DQN (returned by the DQN function) to predict moves
            replay_buffer: A ReplayBuffer object for holding all previous experiences
            n_actions: Number of possible actions for the given environment
            input_shape: Tuple/list describing the shape of the pre-processed environment
            batch_size: Number of samples to draw from the replay memory every updating session
            history_length: Number of historical frames available to the agent
            eps_initial: Initial epsilon value.
            eps_final: The "half-way" epsilon value.  The epsilon value decreases more slowly after this
            eps_final_frame: The final epsilon value
            eps_evaluation: The epsilon value used during evaluation
            eps_annealing_frames: Number of frames during which epsilon will be annealed to eps_final, then eps_final_frame
            replay_buffer_start_size: Size of replay buffer before beginning to learn (after this many frames, epsilon is decreased more slowly)
            max_frames: Number of total frames the agent will be trained for
            use_per: Use PER instead of classic experience replay
        """

        self.n_actions = n_actions
        self.input_shape = input_shape
        self.history_length = history_length

        # Memory information
        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.replay_buffers = replay_buffers
        self.use_per = use_per

        # Epsilon information
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_frame = eps_final_frame
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_frames = eps_annealing_frames

        # Slopes and intercepts for exploration decrease
        # (Credit to Fabio M. Graetz for this and calculating epsilon based on frame number)
        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_frames
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_final - self.eps_final_frame) / (self.max_frames - self.eps_annealing_frames - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final_frame - self.slope_2*self.max_frames

        # DQN
        self.AMN = amn

        # Temperature
        self.T = temperature

        # Set the loss function, either XE or CC
        if loss=='XE':
            self.loss_func = self.XE_loss
        elif loss=='CC':
            self.loss_func = self.CC_loss
        else:
            raise ValueError("Loss function:", loss, "not supported")

    def calc_epsilon(self, frame_number, evaluation=False):
        """Get the appropriate epsilon value from a given frame number
        Arguments:
            frame_number: Global frame number (used for epsilon)
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            The appropriate epsilon value
        """
        if evaluation:
            return self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif frame_number >= self.replay_buffer_start_size and frame_number < self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope*frame_number + self.intercept
        elif frame_number >= self.replay_buffer_start_size + self.eps_annealing_frames:
            return self.slope_2*frame_number + self.intercept_2

    def get_action(self, frame_number, state, actions, evaluation=False):
        """Query the DQN for an action given a state
        Arguments:
            frame_number: Global frame number (used for epsilon)
            state: State to give an action for
            evaluation: True if the model is evaluating, False otherwise (uses eps_evaluation instead of default epsilon value)
        Returns:
            An integer as the predicted move
        """

        # Calculate epsilon based on the frame number
        eps = self.calc_epsilon(frame_number, evaluation)

        # With chance epsilon, take a random action
        if np.random.rand(1) < eps:
            return np.random.choice(actions)

        # Otherwise, query the DQN for an action
        q_vals = self.AMN.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))[0][actions]

        return actions[q_vals.argmax()]

    def get_intermediate_representation(self, state, layer_names=None, stack_state=True):
        """
        Get the output of a hidden layer inside the model.  This will be/is used for visualizing model
        Arguments:
            state: The input to the model to get outputs for hidden layers from
            layer_names: Names of the layers to get outputs from.  This can be a list of multiple names, or a single name
            stack_state: Stack `state` four times so the model can take input on a single (84, 84, 1) frame
        Returns:
            Outputs to the hidden layers specified, in the order they were specified.
        """
        # Prepare list of layers
        if isinstance(layer_names, list) or isinstance(layer_names, tuple):
            layers = [self.AMN.get_layer(name=layer_name).output for layer_name in layer_names]
        else:
            layers = self.AMN.get_layer(name=layer_names).output

        # Model for getting intermediate output
        temp_model = tf.keras.Model(self.AMN.inputs, layers)

        # Stack state 4 times
        if stack_state:
            if len(state.shape) == 2:
                state = state[:, :, np.newaxis]
            state = np.repeat(state, self.history_length, axis=2)

        # Put it all together
        return temp_model.predict(state.reshape((-1, self.input_shape[0], self.input_shape[1], self.history_length)))

    def add_experience(self, game_index, action, frame, reward, terminal, clip_reward=True):
        """Wrapper function for adding an experience to the Agent's replay buffer"""
        self.replay_buffers[game_index].add_experience(action, frame, reward, terminal, clip_reward)

    def learn(self, game_index, expert, action_codes, priority_scale=1.0):
        """Sample a batch and use it to improve the AMN
        Arguments:
            batch_size: How many samples to draw for an update
            gamma: Reward discount
            frame_number: Global frame number (used for calculating importances)
            priority_scale: How much to weight priorities when sampling the replay buffer. 0 = completely random, 1 = completely based on priority
        Returns:
            The loss between the predicted and target Q as a float
        """

        if self.use_per:
            (states, actions, rewards, new_states, terminal_flags), importance, indices = \
                self.replay_buffers[game_index].get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
        else:
            states, actions, rewards, new_states, terminal_flags = \
                self.replay_buffers[game_index].get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)

        # new_states = tf.convert_to_tensor(new_states)
        # terminal_flags = tf.convert_to_tensor(terminal_flags, dtype='float32')
        loss, error = self.mimic_gradients(states, expert, action_codes)

        return loss, error

    # @tf.function
    def mimic_gradients(self, states, expert, action_codes):
        # Implements the 'Policy regression objective' and its gradient
        target_q_vals = expert.DQN(states)
        targets = tf.math.softmax(target_q_vals / self.T, axis=1)

        with tf.GradientTape() as tape:
            q_values = self.AMN(states) / self.T
            # Select only the actions relevant to the current game:
            q_values = tf.gather(q_values, action_codes, axis=1)

            loss = self.loss_func(q_values, targets)

            if self.use_per:
                # Multiply the loss by importance, so that the gradient is also scaled.
                # The importance scale reduces bias against situataions that are sampled
                # more frequently.
                loss = tf.reduce_mean(loss * importance)
            else:
                loss = tf.reduce_mean(loss)

        model_gradients = tape.gradient(loss, self.AMN.trainable_variables)
        self.AMN.optimizer.apply_gradients(zip(model_gradients, self.AMN.trainable_variables))


        return loss, 0.0

    def XE_loss(self, q_values, targets):
        """
        Returns the cross-entropy loss for the AMN model
        """
        return tf.losses.categorical_crossentropy(targets, q_values, from_logits=True)

    def CC_loss(self, q_values, targets):
        """
        Returns the CC loss for the CC-AMN model
        """
        eta = q_values[:, 0:-1] - tf.reshape(q_values[:, -1], [-1, 1])
        temp_mean = tf.stop_gradient(cc_mean(eta))
        discard = tf.math.reduce_any(tf.math.is_nan(temp_mean), axis=1)
        to_keep = tf.reduce_all(tf.math.is_finite(temp_mean), axis=1)
        full = cc_log_prob(targets[to_keep], eta[to_keep])
        partial = - self.XE_loss(q_values[discard], targets[discard])
        print(tf.reduce_any(to_keep))
        print(tf.reduce_sum(tf.cast(to_keep, tf.float32)))
        return - tf.reduce_sum(partial) - tf.reduce_sum(full)


    def save(self, folder_name, **kwargs):
        """Saves the Agent and all corresponding properties into a folder
        Arguments:
            folder_name: Folder in which to save the Agent
            **kwargs: Agent.save will also save any keyword arguments passed.  This is used for saving the frame_number
        """

        # Create the folder for saving the agent
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # Save DQN and target DQN
        self.AMN.save(folder_name + '/dqn.h5')

        # Save meta
        # with open(folder_name + '/meta.json', 'w+') as f:
        #     f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))  # save replay_buffer information and any other information

    def load(self, folder_name, load_replay_buffer=True):
        """Load a previously saved Agent from a folder
        Arguments:
            folder_name: Folder from which to load the Agent
        Returns:
            All other saved attributes, e.g., frame number
        """

        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # Load DQNs
        self.AMN = tf.keras.models.load_model(folder_name + '/dqn.h5')
        self.optimizer = self.AMN.optimizer

        # Load meta
        with open(folder_name + '/meta.json', 'r') as f:
            meta = json.load(f)

        del meta['buff_count'], meta['buff_curr']  # we don't want to return this information
        return meta


# Create environment
if __name__ == "__main__":
    # ------------------------------------------
    # Set up parameters
    # ------------------------------------------
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ENVS', dest='ENVS', type=str, default='default')
    parser.add_argument('--T', dest='T', type=float, default=1.0)
    parser.add_argument('--LOSS', dest='LOSS', type=str, default='XE')

    args = parser.parse_args()
    ENVS = args.ENVS
    T = args.T
    LOSS = args.LOSS
    if ENVS == 'default':
        env_names = [
            'AtlantisDeterministic-v4',
            'BoxingDeterministic-v4',
            'BreakoutDeterministic-v4',
            'CrazyClimberDeterministic-v4',
            'EnduroDeterministic-v4',
            'PongDeterministic-v4',
            'SeaquestDeterministic-v4',
            'SpaceInvadersDeterministic-v4'
        ]
    else:
        env_names = [ENVS]
    num_envs = len(env_names)
    if num_envs == 1:
        TENSORBOARD_DIR = 'amn/tensorboard/' + env_names[0] + '.loss=' + LOSS + '.T=' + str(T) + datetime.now().strftime("%Y-%m-%d-%H%M%S")
        SAVE_PATH = 'amn/saves/' + env_names[0] + '.loss=' + LOSS + '.T=' + str(T)
    else:
        TENSORBOARD_DIR = 'amn/tensorboard/multi' + '.loss=' + LOSS + '.T=' + str(T) + datetime.now().strftime("%Y-%m-%d-%H%M%S")
        SAVE_PATH = 'amn/saves/multi' + '.loss=' + LOSS + '.T=' + str(T)

    game_wrappers = []
    for env_name in env_names:
        game_wrapper = GameWrapper(env_name, MAX_NOOP_STEPS)
        game_wrapper.env.seed(SEED)
        print("The environment has the following {} actions: {}".format(game_wrapper.env.action_space.n, game_wrapper.env.unwrapped.get_action_meanings()))
        game_wrappers.append(game_wrapper)

    all_actions = get_actions(game_wrappers)
    print(all_actions)

    # Code the actions into numbers and set up the experts
    action_codes = []
    inv_action_codes = []
    saved_experts = find_saved_experts(env_names)
    print(saved_experts)
    experts = []
    replay_buffers = []
    for i in range(num_envs):
        game_actions = game_wrappers[i].env.unwrapped.get_action_meanings()
        action_code = [all_actions.index(action) for action in game_actions]
        action_codes.append(action_code)
        inv_action_code = [game_actions.index(action) for action in all_actions if action in game_actions]
        inv_action_codes.append(inv_action_code)
        MAIN_DQN = build_q_network(len(game_actions), input_shape=INPUT_SHAPE)
        TARGET_DQN = build_q_network(len(game_actions), input_shape=INPUT_SHAPE)
        replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
        expert = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, len(game_actions), input_shape=INPUT_SHAPE,
                       batch_size=BATCH_SIZE, use_per=USE_PER)
        expert.load(saved_experts[i], False)
        experts.append(expert)
        replay_buffers.append(replay_buffer)

    print(action_codes)

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)


    # Build main and target networks
    MAIN_AMN = build_q_network(len(all_actions), LEARNING_RATE, input_shape=INPUT_SHAPE)

    amn_agent = AMN(MAIN_AMN, replay_buffers, len(all_actions), temperature=T, loss=LOSS, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, use_per=USE_PER)

    frame_number = 0
    rewards = []
    loss_list = []

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # We do a training/evaluation cycle for each game
                for game_index in range(num_envs):
                    env_name = env_names[game_index]
                    start_time = time.time()
                    # Pick a game:
                    game_wrapper = game_wrappers[game_index]
                    expert = experts[game_index]
                    epoch_frame = 0
                    while epoch_frame < FRAMES_BETWEEN_EVAL:
                        game_wrapper.reset()
                        life_lost = True
                        episode_reward_sum = 0
                        for _ in range(MAX_EPISODE_LENGTH):
                            # Get action
                            action = amn_agent.get_action(frame_number, game_wrapper.state, action_codes[game_index])

                            # Take step
                            processed_frame, reward, terminal, life_lost = game_wrapper.step(action_codes[game_index].index(action))
                            frame_number += 1
                            epoch_frame += 1
                            episode_reward_sum += reward

                            # Add experience to replay memory
                            amn_agent.add_experience(game_index, action=action,
                                                frame=processed_frame[:, :, 0],
                                                reward=reward, clip_reward=CLIP_REWARD,
                                                terminal=life_lost)

                            # Update agent
                            if frame_number % UPDATE_FREQ == 0 and amn_agent.replay_buffers[game_index].count > MIN_REPLAY_BUFFER_SIZE:
                                # need to pass the frame number as tf object to avoid @tf.funcion retracing
                                # end_time = time.time()
                                # print('env time:', end_time - start_time)
                                # start_time = time.time()
                                loss, _ = amn_agent.learn(game_index, expert, action_codes[game_index], priority_scale=PRIORITY_SCALE)
                                loss_list.append(float(loss.numpy()))
                                # end_time = time.time()
                                # print("gradient time:", end_time - start_time)
                                # start_time = time.time()

                            # Break the loop when the game is over
                            if terminal:
                                terminal = False
                                break

                        rewards.append(episode_reward_sum)

                        # Output the progress every 10 games
                        if len(rewards) % 10 == 0:
                            # Write to TensorBoard
                            if WRITE_TENSORBOARD:
                                tf.summary.scalar(env_name + '-Reward', np.mean(rewards[-10:]), frame_number)
                                tf.summary.scalar(env_name + '-Loss', np.mean(loss_list[-100:]), frame_number)
                                writer.flush()

                            print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')
                            start_time = time.time()

                    # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                    terminal = True
                    eval_rewards = []
                    evaluate_frame_number = 0

                    for _ in range(EVAL_LENGTH):
                        if terminal:
                            game_wrapper.reset(evaluation=True)
                            life_lost = True
                            episode_reward_sum = 0
                            terminal = False

                        # Breakout requires a "fire" action (action #1) to start the
                        # game each time a life is lost.
                        # Otherwise, the agent would sit around doing nothing.
                        if life_lost and env_name[0:8] == 'Breakout':
                            action = 1
                        else:
                            action = amn_agent.get_action(frame_number, game_wrapper.state, action_codes[game_index], evaluation=True)

                        # Step action
                        _, reward, terminal, life_lost = game_wrapper.step(action_codes[game_index].index(action))
                        evaluate_frame_number += 1
                        episode_reward_sum += reward

                        # On game-over
                        if terminal:
                            eval_rewards.append(episode_reward_sum)

                    if len(eval_rewards) > 0:
                        final_score = np.mean(eval_rewards)
                    else:
                        # In case the game is longer than the number of frames allowed
                        final_score = episode_reward_sum
                    # Print score and write to tensorboard
                    print('Evaluation score:', final_score, 'Time taken:', time.time() - start_time)
                    if WRITE_TENSORBOARD:
                        tf.summary.scalar(env_name + '-EvaluationScore', final_score, frame_number)
                        writer.flush()

                    # Save model
                    if len(rewards) > 300 and SAVE_PATH is not None:
                        amn_agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None:
            print('Saving...')
            amn_agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
            print('Saved.')
