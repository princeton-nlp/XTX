# Built-in Imports
from typing import Union, List

# Libraries

# Custom imports
from utils.util import get_name_from_path
from utils.env import JerichoEnv
from utils.vec_env import VecEnv

OBJECTS_DIR = './saved_objects'


class Trainer:
    """General trainer class.
    """

    def __init__(self, tb, log, agent, envs, eval_env, args):
        self.tb = tb
        self.log = log
        self.agent = agent
        self.envs = envs
        self.eval_env = eval_env

        self.max_steps = args.max_steps
        self.log_freq = args.log_freq
        self.target_update_freq = args.target_update_freq
        self.q_update_freq = args.q_update_freq
        self.checkpoint_freq = args.checkpoint_freq
        self.eval_freq = args.eval_freq
        self.batch_size = args.batch_size
        self.game = get_name_from_path(args.rom_path)
        self.eps = args.eps
        self.eps_type = args.eps_type
        self.dynamic_episode_length = args.dynamic_episode_length

        self.steps = 0

    def train(self):
        """Trains the agent.

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError

    def setup_env(self):
        """Setup the environment.

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError

    def update_envs(self):
        """Step through all the envs.

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError

    def end_step(self, step: int):
        """Perform any logging, saving, evaluation, etc.
        that happens at the end of each step.

        Args:
            step (int): the current step number

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError

    def update_agent(self):
        """Update the agent with gradient descent.
        """
        # Update
        loss = self.agent.update()

        # Log the loss
        if loss is not None:
            self.tb.logkv_mean('Loss', loss)

    def evaluate(self, nb_episodes: int = 3):
        """Evaluate the agent on several runs of the episodes and return the average reward.

        Args:
            nb_episodes (int, optional): number of episodes to average over. Defaults to 3.

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError

    def write_to_logs(self, step: int, start: float, envs: Union[List[JerichoEnv], VecEnv], max_score: int):
        """Write to loggers.

        Args:
            step (int): current step
            start (float): time at start of training
            envs (Union[List[JerichoEnv], VecEnv]): collections of environments
            max_score (int): maximum score seen so far

        Raises:
            NotImplementedError: implemented by child class
        """
        raise NotImplementedError
