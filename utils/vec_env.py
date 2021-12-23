# Built-in imports 
from multiprocessing import Process, Pipe, Manager

# Libraries
import numpy as np


def worker(remote, parent_remote, env):
    parent_remote.close()
    try:
        done = False
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if done:
                    ob, info = env.reset()
                    reward = 0
                    done = False
                else:
                    ob, reward, done, info = env.step(data)
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob, info = env.reset()
                done = False
                remote.send((ob, info))
            elif cmd == 'get_ngram_hits':
                remote.send(env.ngram_hits)
            elif cmd == 'get_end_scores':
                remote.send(env.get_end_scores(last=100))
            elif cmd == 'get_current_score':
                remote.send(env.get_score())
            elif cmd == 'get_current_step':
                remote.send(env.steps)
            elif cmd == 'add_traj':
                env.traj.append(data)
                remote.send(env.traj)
            elif cmd == 'add_full_traj':
                env.full_traj.append(data)
                remote.send(env.full_traj)
            elif cmd == 'update_ngram_hits':
                env.ngram_hits += data
                remote.send(env.ngram_hits)
            elif cmd == 'set_env_limit':
                env.step_limit = data
                remote.send(env.step_limit)
            elif cmd == 'turn_off_trajectory':
                env.turn_off_trajectory()
                remote.send(True)
            elif cmd == 'get_trajectory_state':
                traj_state = env.get_trajectory_state()
                remote.send(traj_state)
            elif cmd == 'get_env_limit':
                remote.send(env.step_limit)
            elif cmd == "get_traj":
                remote.send(env.traj)
            elif cmd == 'get_ngram_needs_update':
                remote.send(env.ngram_needs_update)
            elif cmd == 'set_ngram_needs_update':
                env.ngram_needs_update = data
                remote.send(env.ngram_needs_update)
            elif cmd == 'get_cache_size':
                remote.send(len(env.cache))
            elif cmd == 'get_unique_acts_size':
                remote.send(len(env.unique_acts))
            elif cmd == 'close':
                env.close()
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


class VecEnv:
    def __init__(self, num_envs, env):
        self.closed = False
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(num_envs)])
        env.cache = Manager().dict()
        env.unique_acts = Manager().dict()
        self.ps = [Process(target=worker, args=(work_remote, remote, env))
                   for (work_remote, remote) in zip(self.work_remotes, self.remotes)]
        for p in self.ps:
            # p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def __len__(self):
        return self.num_envs

    def step(self, actions):
        self._assert_not_closed()
        assert len(
            actions) == self.num_envs, "Error: incorrect number of actions."
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rewards, dones, infos = zip(*results)
        return list(obs), np.stack(rewards), list(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def get_ngram_hits(self, i):
        self._assert_not_closed()
        self.remotes[i].send(('get_ngram_hits', None))
        ngram_hits = self.remotes[i].recv()
        return ngram_hits

    def update_ngram_hits(self, beta_vec):
        self._assert_not_closed()
        for i, remote in enumerate(self.remotes):
            remote.send(('update_ngram_hits', beta_vec[i]))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def turn_off_trajectory(self, i: int):
        self._assert_not_closed()
        self.remotes[i].send(('turn_off_trajectory', None))
        result = self.remotes[i].recv()
        return result

    def get_trajectory_state(self, i: int):
        self._assert_not_closed()
        self.remotes[i].send(('get_trajectory_state', None))
        result = self.remotes[i].recv()
        return result

    def reset_one(self, i):
        self._assert_not_closed()
        self.remotes[i].send(('reset', None))
        ob, info = self.remotes[i].recv()
        return ob, info

    def get_end_scores(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_end_scores', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def get_current_scores(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_current_score', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def set_env_limit(self, limit, i):
        self._assert_not_closed()
        self.remotes[i].send(('set_env_limit', limit))
        result = self.remotes[i].recv()
        return result

    def get_env_limit(self):
        self._assert_not_closed()
        self.remotes[0].send(('get_env_limit', None))
        limit = self.remotes[0].recv()
        return limit

    def get_ngram_needs_update(self, i: int):
        self._assert_not_closed()
        self.remotes[i].send(('get_ngram_needs_update', None))
        result = self.remotes[i].recv()
        return result

    def set_ngram_needs_update(self, update):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('set_ngram_needs_update', update))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def set_ngram_needs_update_i(self, update: bool, i: int):
        self._assert_not_closed()
        self.remotes[i].send(('set_ngram_needs_update', update))
        result = self.remotes[i].recv()
        return result

    def get_current_steps(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_current_step', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def get_cache_size(self):
        self._assert_not_closed()
        self.remotes[0].send(('get_cache_size', None))
        result = self.remotes[0].recv()
        return result

    def get_unique_acts(self):
        self._assert_not_closed()
        self.remotes[0].send(('get_unique_acts_size', None))
        result = self.remotes[0].recv()
        return result

    def add_traj(self, action_strs):
        self._assert_not_closed()
        for remote, action_str in zip(self.remotes, action_strs):
            remote.send(('add_traj', action_str))
        results = [remote.recv() for remote in self.remotes]
        return results

    def add_full_traj(self, traj_steps):
        self._assert_not_closed()
        for remote, traj_step in zip(self.remotes, traj_steps):
            remote.send(('add_full_traj', traj_step))
        results = [remote.recv() for remote in self.remotes]
        return results

    def add_full_traj_i(self, i: int, traj_step):
        self._assert_not_closed()
        self.remotes[i].send(('add_full_traj', traj_step))
        result = self.remotes[i].recv()
        return result

    def get_traj_i(self, i: int):
        self._assert_not_closed()
        self.remotes[i].send(('get_traj', None))
        result = self.remotes[i].recv()
        return result

    def close_extras(self):
        self.closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"
