import numpy as np
import gym
from gym import spaces

class CustomerRiskEnv(gym.Env):
    def __init__(self):
        super(CustomerRiskEnv, self).__init__()

        # States: 0=Good, 1=At-Risk, 2=Delinquent, 3=Default, 4=Closed
        self.states = ['Good', 'At-Risk', 'Delinquent', 'Default', 'Closed']
        self.num_states = len(self.states)

        # Actions: 0=Do nothing, 1=Warn, 2=Freeze account
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.num_states)

        self.transition_matrix = np.array([
            [0.90, 0.07, 0.01, 0.00, 0.02],
            [0.10, 0.70, 0.15, 0.02, 0.03],
            [0.00, 0.10, 0.65, 0.20, 0.05],
            [0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ])

        self.state = 0  # start in 'Good'

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        done = False
        reward = 0

        # Custom reward logic
        if self.state == 2 and action == 2:
            reward = 10  # freeze Delinquent = good
        elif self.state == 2 and action == 0:
            reward = -20  # ignore Delinquent = risky
        elif self.state == 1 and action == 1:
            reward = 5  # warn At-Risk = good
        elif self.state == 3:
            reward = -100
            done = True
        elif self.state == 4:
            reward = -40
            done = True

        # Transition to next state
        next_state = np.random.choice(self.num_states, p=self.transition_matrix[self.state])
        self.state = next_state

        return self.state, reward, done, {}

# Sample run
if __name__ == "__main__":
    env = CustomerRiskEnv()
    obs = env.reset()
    print(f"Initial State: {env.states[obs]}")

    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        print(f"Action: {action}, Next State: {env.states[next_obs]}, Reward: {reward}")
        if done:
            break

import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# Collect run log
log = [f"Initial State: {env.states[obs]}"]
env = CustomerRiskEnv()
obs = env.reset()

for _ in range(10):
    action = env.action_space.sample()
    next_obs, reward, done, _ = env.step(action)
    log.append(f"Action: {action}, Next State: {env.states[next_obs]}, Reward: {reward}")
    if done:
        break

def show_rl_popup(log_lines):
    root = tk.Tk()
    root.title("RL Environment Run Log")

    text_widget = ScrolledText(root, wrap=tk.WORD, width=80, height=20)
    text_widget.insert(tk.END, "\n".join(log_lines))
    text_widget.pack(padx=10, pady=10)
    
    root.mainloop()

show_rl_popup(log)
