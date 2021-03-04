import gym
import retro
import argparse
import numpy as np
import os
from datetime import datetime
from game_models.ddqn_game_model import DDQNTrainer, DDQNSolver
from Wrapper.kaboom_wrappers import KaboomWrapper
from Discretizer import KaboomDiscretizer, TutankhamDiscretizer, TennisDiscretizer, StarGunnerDiscretizer
from Wrapper.gym_wrappers import MainGymWrapper
from lineNotification import LineNotification

FRAMES_IN_OBSERVATION = 4
FRAME_SIZE = 84
INPUT_SHAPE = (FRAMES_IN_OBSERVATION, FRAME_SIZE, FRAME_SIZE)


class Main:

    def __init__(self):
        game_name, game_mode, render, total_step_limit, total_episode_limit, clip = self._args()
        #game_name = "Kaboom-Atari2600"
        record_folder = "output/record/" + game_name + "/" + datetime.now().strftime('%Y-%m-%d_%H-%M')
        if not os.path.exists(record_folder):
            os.makedirs(record_folder)
        if game_name == "Kaboom-Atari2600":
            env = MainGymWrapper.wrap(KaboomDiscretizer(retro.make(game_name, record=record_folder + "/.")))
        elif game_name == "Tutankham-Atari2600":
            env = MainGymWrapper.wrap(TutankhamDiscretizer(retro.make(game_name, record=record_folder + "/.")))
        elif game_name == "Tennis-Atari2600":
            env = MainGymWrapper.wrap(TennisDiscretizer(retro.make(game_name, record=record_folder + "/.")))
        elif game_name == "StarGunner-Atari2600":
            env = MainGymWrapper.wrap(StarGunnerDiscretizer(retro.make(game_name, record=record_folder + "/.")))
        print("Game name: " + game_name)
        print("Amount of actions:" + str(env.action_space))
        print("Observation Type: " + str(env.observation_space))
        self._main_loop(self._game_model(game_mode, game_name, env.action_space.n), env, render, total_step_limit, total_episode_limit, clip)
        
        # Upper Passed #
        
    def _main_loop(self, game_model, env, render, total_step_limit, total_episode_limit, clip):
        """A main loop for Reinforcement Learning"""
        run = 0
        total_step = 0
        while True:
            if total_episode_limit is not None and run >= total_episode_limit:
                print("Reached total run limit of: " + str(total_episode_limit))
                exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    LineNotification.networking()
                    exit(0)
                total_step += 1
                step += 1

                if render:
                    env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                
                if clip:
                    np.sign(reward)
                
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state
                game_model.step_update(total_step)

                if terminal:
                    print(total_step)
                    game_model.save_run(score, step, run)
                    break
    
    def _args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-g", "--game", help="Choose available game Defualt is Kaboom'.", default="Kaboom-Atari2600")
        parser.add_argument("-m", "--mode", help="Choose from available modes: ddqn_train, ddqn_test Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-r", "--render", help="Choose if the game should be rendered. Default is 'False'.", default=False, type=bool)
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-tel", "--total_episode_limit", help="Choose after how many Episodes we should stop. Default is None (no limit).", default=None, type=int)
        parser.add_argument("-c", "--clip", help="Choose whether we should clip rewards to (0, 1) range. Default is 'True'", default=True, type=bool)
        args = parser.parse_args()
        game_name = args.game
        game_mode = args.mode
        render = args.render
        total_step_limit = args.total_step_limit
        total_episode_limit = args.total_episode_limit
        clip = args.clip
        print("Your Mode is: " + str(game_mode))
        print("Allow rendering: " + str(render))
        print("Allow Clipping:" + str(clip))
        print("Total Episode(s) limit: " + str(total_episode_limit))
        print("Total Step(s) limit: " + str(total_step_limit))

        return game_name, game_mode, render, total_step_limit, total_episode_limit, clip


    def _game_model(self, game_mode,game_name, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(game_name, INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNSolver(game_name, INPUT_SHAPE, action_space)
        else:
            print("Unrecognized mode. Use --help")
            exit(1)
            
if __name__ == "__main__":
    Main()
