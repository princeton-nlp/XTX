# Built-in imports
import random

# Libraries
import numpy as np
from jericho import *

GAME_DIR = "./games"

def main():
    """Simple setup to be able to easily play around with Jericho env.
    """
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    env = FrotzEnv("{}/{}".format(GAME_DIR, 'zork1.z5'), seed=seed)
    obs, info = env.reset()
    print(obs)
    print("Valid acts:", env.get_valid_actions())
    total = info['score']

    i = 0
    done = False
    buffer = env.get_walkthrough()
    while True:
        if i >= len(buffer):
            act = input()
        else:
            act = buffer[i].strip()

        observation, reward, done, info = env.step(act)

        # state = env.get_state()
        # inv, _, _, _ = env.step('inventory')
        # env.set_state(state)

        # state = env.get_state()
        # loc, _, _, _ = env.step('look')
        # env.set_state(state)

        print("(reward: {})".format(reward) if reward > 0 else "")

    
        print("Action: {}".format(act))
        print("Reward: {}".format(reward))
        print("Obs: {}".format(observation))
        print("Valid acts: {}".format(env.get_valid_actions()))

        total += reward
        i += 1

        print("SCORE: {}".format(total))

if __name__ == "__main__":
    main()
