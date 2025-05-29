from App.car_controller import CarControllerKinematic
from MachineLearning.q_learning import CustomModelQLearning
from MachineLearning.pytorch_dqn import PyTorchDQN

import sys
sys.path.append(".")
from Utils import global_settings as gs


class AutonomousDrivingController(CarControllerKinematic):
    # base that doesnt include DQN
    def __init__(self, state_n):
        super().__init__()
        self.state = [0 for _ in range(state_n)]

        self.brake_amount = 0.5
        self.accelerate_amount = 0.5
        self.steer_amount = 80

        self.max_velocity = 6
        self.max_steering = 80
        self.start_velocity = 3

        self.distance_travelled = 0

        self.current_action = 0
        self.ai_dead = False
        self.prev_state = None

    def end_of_episode(self, verbose=2):
        raise NotImplementedError

    def end_of_frame(self):
        raise NotImplementedError

    def evaluate_reward(self):
        raise NotImplementedError


class AutonomousDrivingControllerCombined(AutonomousDrivingController):
    # driving controller with combined gas and steering networks
    def __init__(self, state_n):
        super().__init__(state_n)
        # Use PyTorch DQN instead of custom model
        self.q_learning = PyTorchDQN(state_n, 5, gs.LOAD_MODEL)  # 5 actions (none, left, right, brake, accelerate)

    def end_of_episode(self, verbose=2):
        # New episode so reset controls
        self.steering_angle = 0
        self.velocity = self.start_velocity

        # Start training and decay probability
        if gs.Q_LEARNING_SETTINGS["TRAINING"]:
            self.q_learning.decay_exploration_probability()
            self.q_learning.train(verbose)

        self.distance_travelled = 0
        self.ai_dead = False
        self.prev_state = None

    def update_transform(self):
        """
        0: nothing
        1: right steer
        2: left steer
        3: brake
        4: accelerate
        """
        # Store previous state for experience replay
        self.prev_state = self.state.copy()

        self.distance_travelled += self.velocity

        action, _ = self.q_learning.get_action(self.state)

        # Q learning actions
        if action == 0:
            self.steering_angle = 0
            self.current_action = 0

        elif action == 1:
            self.steering_angle = self.max_steering
            self.current_action = 1

        elif action == 2:
            self.steering_angle = -self.max_steering
            self.current_action = 2

        elif action == 3:
            self.velocity -= self.brake_amount

            if self.velocity < 0:
                self.velocity = 0
            self.current_action = 3

        elif action == 4:
            self.velocity += self.accelerate_amount

            if self.velocity > self.max_velocity:
                self.velocity = self.max_velocity
            self.current_action = 4

        self.acceleration = 0
        super().update_transform()

    def end_of_frame(self):
        if gs.Q_LEARNING_SETTINGS["TRAINING"]:
            reward = self.evaluate_reward()
            # Update with current state, action, reward, next state and done
            self.q_learning.update_experience_buffer(
                self.prev_state if self.prev_state is not None else self.state, 
                self.current_action, 
                reward, 
                self.state,
                self.ai_dead
            )

    def evaluate_reward(self):
        if self.ai_dead:
            return -1
        
        # Base reward based on velocity
        v = self.velocity/900
        base_reward = v
        
        # Additional rewards based on actions
        action_rewards = {
            0: 0.1,  # small reward for doing nothing (preserves energy)
            1: 0.0,  # no additional reward for steering right
            2: 0.0,  # no additional reward for steering left
            3: -0.1, # small penalty for braking (uses energy)
            4: 0.05  # small reward for acceleration (moving forward)
        }
        
        return base_reward + action_rewards.get(self.current_action, 0)
