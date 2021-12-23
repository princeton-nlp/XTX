# Built-in imports

# Libraries
from jericho import *
from jericho.util import *
from jericho.defines import *

# Custom imports

class JerichoEnv:
    ''' Returns valid actions at each step of the game. '''

    def __init__(self,
                 rom_path,
                 step_limit=None,
                 get_valid=True,
                 cache=None,
                 seed=None,
                 start_from_reward=0,
                 start_from_wt=0,
                 log=None,
                 args=None):
        self.rom_path = rom_path
        self.env = FrotzEnv(rom_path, seed=seed)
        self.bindings = self.env.bindings
        self.steps = 0
        self.step_limit = step_limit
        self.get_valid = get_valid
        self.max_score = 0
        self.end_scores = []
        self.cache = cache
        self.traj = []
        self.full_traj = []
        self.on_trajectory = True
        self.start_from_reward = start_from_reward
        self.start_from_wt = start_from_wt

        self.log = log
        self.cache_hits = 0
        self.ngram_hits = 0
        self.ngram_needs_update = False
        self.filter_drop_acts = args.filter_drop_acts
        self.args = args

    def get_objects(self):
        desc2objs = self.env._identify_interactive_objects(
            use_object_tree=False)
        obj_set = set()
        for objs in desc2objs.values():
            for obj, pos, source in objs:
                if pos == 'ADJ':
                    continue
                obj_set.add(obj)
        return list(obj_set)

    def _get_state_hash(self, ob):
        return self.env.get_world_state_hash()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        # return ob, reward, done, info

        # Initialize with default values
        info['look'] = 'unknown'
        info['inv'] = 'unknown'
        info['valid'] = ['wait', 'yes', 'no']
        if not done:
            save = self.env.get_state()
            hash_save = self._get_state_hash(ob)
            if self.cache is not None and hash_save in self.cache:
                info['look'], info['inv'], info['valid'] = self.cache[
                    hash_save]
                self.cache_hits += 1
            else:
                look, _, _, _ = self.env.step('look')
                info['look'] = look.lower()
                self.env.set_state(save)
                inv, _, _, _ = self.env.step('inventory')
                info['inv'] = inv.lower()
                self.env.set_state(save)
                if self.get_valid:
                    valid = self.env.get_valid_actions()
                    if len(valid) == 0:
                        valid = ['wait', 'yes', 'no']
                    info['valid'] = valid
                if self.cache is not None:
                    self.cache[hash_save] = info['look'], info['inv'], info[
                        'valid']

        self.steps += 1
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        self.max_score = max(self.max_score, info['score'])
        if done:
            self.end_scores.append(info['score'])
        return ob, reward, done, info

    def reset(self):
        initial_ob, info = self.env.reset()

        rewards_encountered = 0
        walkthrough = self.env.get_walkthrough()

        for act in walkthrough:
            if rewards_encountered >= self.start_from_reward:
                break
            initial_ob, reward, _, info = self.env.step(act)
            if reward > 0:
                rewards_encountered += 1

        for act in walkthrough[:self.start_from_wt]:
            initial_ob, reward, _, info = self.env.step(act)

        save = self.env.get_state()
        look, _, _, _ = self.env.step('look')
        info['look'] = look
        self.env.set_state(save)
        inv, _, _, _ = self.env.step('inventory')
        info['inv'] = inv
        self.env.set_state(save)
        valid = self.env.get_valid_actions()
        info['valid'] = valid
        self.steps = 0
        self.max_score = 0
        self.ngram_hits = 0
        self.traj = []
        self.full_traj = []
        self.on_trajectory = True
        return initial_ob, info

    def turn_off_trajectory(self):
        self.on_trajectory = False

    def get_trajectory_state(self):
        return self.on_trajectory

    def get_dictionary(self):
        if not self.env:
            self.create()
        return self.env.get_dictionary()

    def get_action_set(self):
        return None

    def get_end_scores(self, last=1):
        last = min(last, len(self.end_scores))
        return sum(self.end_scores[-last:]) / last if last else 0

    def close(self):
        self.env.close()

    def get_walkthrough(self):
        return self.env.get_walkthrough()

    def get_score(self):
        return self.env.get_score()
