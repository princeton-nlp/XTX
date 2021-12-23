# Built-in imports
from typing import List

# Libraries
from tqdm import tqdm

# Custom imports
from utils.util import inv_process_action, process_action, load_object
from utils.env import JerichoEnv

from scripts.train_rl import parse_args

GAME_DIR = "./games"

def is_act_missing_in_wt(candidates: List[str], act: str, env):
    """
    Check if wt action is in Jericho candidates.
    """
    state = env.get_state()
    env.step(act)
    wt_diff = str(env._get_world_diff())
    env.set_state(state)
    if wt_diff == '((), (), (), ())':
        return False
        # return process_action(act) not in list(map(lambda x: process_action(x), candidates))

    candidate_diffs = []
    for candidate in candidates:
        env.set_state(state)
        env.step(candidate)
        candidate_diffs.append(str(env._get_world_diff()))
    env.set_state(state)

    gold_acts = []
    for can_diff, can_act in zip(candidate_diffs, candidates):
        if can_diff == wt_diff:
            gold_acts.append(can_act)
            break

    return len(gold_acts) == 0


def get_missing_wt_acts(game: str):
    """
    Get missing wt acts.
    """

    cache = dict()
    args = parse_args()
    env = JerichoEnv("{}/{}".format(GAME_DIR, game), cache=cache, args=args)
    ob, info = env.reset()

    missing_wt = set()
    walkthrough = env.get_walkthrough()
    for i, act in tqdm(enumerate(walkthrough), desc='Getting missing wt acts ...'):
        candidates = info['valid']
        missing = is_act_missing_in_wt(candidates, act, env.env)
        if missing:
            missing_wt = missing_wt.union(
                {process_action(act), inv_process_action(act)})

        next_ob, reward, done, info = env.step(act)

        ob = next_ob

    print("Cache hits: {}".format(env.cache_hits))
    return missing_wt

def main():
    games = ["zork1.z5"]
    for game in games:
        missing_acts = get_missing_wt_acts(game)
        with open('./missing_wt_acts_{}.txt'.format(game), 'w') as f:
            for act in missing_acts:
                f.write('{};'.format(act))


if __name__ == "__main__":
    main()
