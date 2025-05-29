import numpy as np
import random
import pygame
import sys
import os

# Set seed for reproducibility
np.random.seed(1)
random.seed(1)

# Initialize pygame
pygame.init()

# Add current directory to path for easier imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from App.world import World, Map
from App.AppScreens.ai_car_simulation import run_ai_car_simulation
from App.AppScreens.map_selection import run_map_selection
from Utils import global_settings as gs


def main_app(map_override=None, background=False, end_at_min_epsilon=False, verbose=2):
    """
    Main application entry point
    
    :param map_override: a map name to use instead of showing map selection screen, defaults to None
    :param background: run the PyGame window in the background, or use dimensions from settings
    :param end_at_min_epsilon: close training window when min epsilon is reached
    :param verbose: verbosity level for training output
    :return: DQN object with learned parameters
    """
    params = [map_override, background, end_at_min_epsilon]
    if any(params) and not all(params):
        raise TypeError("Run in background without required parameters. Impossible to choose map/end.")

    # Create Window
    # if in background, use a surface to simulate screen
    screen = pygame.Surface((1, 1)) if background else pygame.display.set_mode((gs.WIDTH, gs.HEIGHT))

    # ================================================= MAP BUILDER ====================================================
    if map_override is None:
        selected_map = run_map_selection(screen)
    else:
        selected_map = Map.load_map(map_override)[0]

    world = World(selected_map)
    world.replicate_map_spawn()

    # ================================================ GAME LOOP =======================================================
    run_ai_car_simulation(screen, world, end_at_min_epsilon=end_at_min_epsilon, verbose=verbose)

    # Show reward graphs, error graphs and save models, IF TRAINING
    # Return the DQN model
    controller = world.ai_car.controller
    if gs.Q_LEARNING_SETTINGS["TRAINING"] and not background:
        controller.q_learning.reward_graph()
        controller.q_learning.error_graph(color="red")
        controller.q_learning.save_model("pytorch_dqn")
    return controller.q_learning


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run DQN Car Simulation')
    parser.add_argument('--background', action='store_true', help='Run in background mode')
    parser.add_argument('--map', type=str, help='Map name to use')
    parser.add_argument('--min-epsilon', action='store_true', help='End at min epsilon')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity level (0-3)')
    args = parser.parse_args()
    
    # Check if we're running in background mode
    if args.background and args.map:
        q = main_app(background=True, map_override=args.map, end_at_min_epsilon=args.min_epsilon, verbose=args.verbose)
        q.reward_graph()
        q.error_graph(color="red")
        q.save_model("pytorch_dqn")
    else:
        # Run the main app with default settings
        main_app(verbose=args.verbose) 