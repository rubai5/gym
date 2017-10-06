import logging
import pkg_resources
import re
from gym import error
import random
import warnings

logger = logging.getLogger(__name__)
# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2
#
# 2016-10-31: We're experimentally expanding the environment ID format
# to include an optional username.
env_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')

def load(name):
    entry_point = pkg_resources.EntryPoint.parse('x={}'.format(name))
    result = entry_point.load(False)
    return result

class EnvSpec(object):
    """A specification for a particular instance of the environment. Used
    to register the parameters for official evaluations.

    Args:
        id (str): The official environment ID
        entry_point (Optional[str]): The Python entrypoint of the environment class (e.g. module.name:Class)
        trials (int): The number of trials to average reward over
        reward_threshold (Optional[int]): The reward threshold before the task is considered solved
        local_only: True iff the environment is to be used only on the local machine (e.g. debugging envs)
        kwargs (dict): The kwargs to pass to the environment class
        nondeterministic (bool): Whether this environment is non-deterministic even after seeding
        tags (dict[str:any]): A set of arbitrary key-value tags on this environment, including simple property=True tags

    Attributes:
        id (str): The official environment ID
        trials (int): The number of trials run in official evaluation
    """

    def __init__(self, id, entry_point=None, trials=100, reward_threshold=None, local_only=False, kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None, max_episode_seconds=None, timestep_limit=None):
        self.id = id
        # Evaluation parameters
        self.trials = trials
        self.reward_threshold = reward_threshold
        # Environment properties
        self.nondeterministic = nondeterministic

        if tags is None:
            tags = {}
        self.tags = tags

        # BACKWARDS COMPAT 2017/1/18
        if tags.get('wrapper_config.TimeLimit.max_episode_steps'):
            max_episode_steps = tags.get('wrapper_config.TimeLimit.max_episode_steps')
            # TODO: Add the following deprecation warning after 2017/02/18
            # warnings.warn("DEPRECATION WARNING wrapper_config.TimeLimit has been deprecated. Replace any calls to `register(tags={'wrapper_config.TimeLimit.max_episode_steps': 200)}` with `register(max_episode_steps=200)`. This change was made 2017/1/31 and is included in gym version 0.8.0. If you are getting many of these warnings, you may need to update universe past version 0.21.3")

        tags['wrapper_config.TimeLimit.max_episode_steps'] = max_episode_steps
        ######

        # BACKWARDS COMPAT 2017/1/31
        if timestep_limit is not None:
            max_episode_steps = timestep_limit
            # TODO: Add the following deprecation warning after 2017/03/01
            # warnings.warn("register(timestep_limit={}) is deprecated. Use register(max_episode_steps={}) instead.".format(timestep_limit, timestep_limit))
        ######

        self.max_episode_steps = max_episode_steps
        self.max_episode_seconds = max_episode_seconds

        # We may make some of these other parameters public if they're
        # useful.
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to register malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id, env_id_re.pattern))
        self._env_name = match.group(1)
        self._entry_point = entry_point
        self._local_only = local_only
        self._kwargs = {} if kwargs is None else kwargs

    def make(self):
        """Instantiates an instance of the environment with appropriate kwargs"""
        print("calling make function in EnvSpecs class", "with environment id", self._entry_point, type(self._entry_point))
        if self._entry_point is None:
            raise error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))

        elif callable(self._entry_point):
            env = self._entry_point()
        else:
            cls = load(self._entry_point)
            env = cls(**self._kwargs)

        # Make the enviroment aware of which spec it came from.
        env.unwrapped._spec = self

        return env


    def check_args(self, names_and_args):
        arg_names = ["K", "potential", "unif_prob", "geo_prob", "diverse_prob", "state_unif_prob",
                      "high_one_prob",
                    "adverse_set_prob", "disj_supp_prob", "geo_high", "unif_high", "geo_ps", "hash_states"]
        arg_types = [int, float, float, float, float, float, float, float, float, int, int, list, dict]
        args = []
        for i, arg in enumerate(names_and_args):
            assert arg[0] == arg_names[i], "Name doesn't match!"
            assert type(arg[1]) == arg_types[i], "Types don't match!"
            args.append(arg[1])
        
        return args


    def make_erdos(self, names_and_args):
        """Instantiates an instance of erdos environment with arguments that were given"""
        
        cls = load(self._entry_point)

        # Check that args are correct and then use
        args = self.check_args(names_and_args)
        env = cls(*args)

        # Make the enviroment aware of which spec it came from.
        env.unwrapped._spec = self

        return env


    def __repr__(self):
        return "EnvSpec({})".format(self.id)

    @property
    def timestep_limit(self):
        return self.max_episode_steps

    @timestep_limit.setter
    def timestep_limit(self, value):
        self.max_episode_steps = value


class EnvRegistry(object):
    """Register an env by ID. IDs remain stable over time and are
    guaranteed to resolve to the same environment dynamics (or be
    desupported). The goal is that results on a particular environment
    should always be comparable, and not depend on the version of the
    code that was running.
    """

    def __init__(self):
        self.env_specs = {}

    def make(self, id, names_and_args=None):
        logger.info('Making new env: %s', id)
        spec = self.spec(id)
        if id == "ErdosGame-v0":
            assert names_and_args != None, "Names and args cannot be None for ErdosGame"
            env = spec.make_erdos(names_and_args)
        else:
            env = spec.make()
        if (env.spec.timestep_limit is not None) and not spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env,
                            max_episode_steps=env.spec.max_episode_steps,
                            max_episode_seconds=env.spec.max_episode_seconds)
        return env

    def all(self):
        return self.env_specs.values()

    def spec(self, id):
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), env_id_re.pattern))

        try:
            return self.env_specs[id]
        except KeyError:
            # Parse the env name and check to see if it matches the non-version
            # part of a valid env (could also check the exact number here)
            env_name = match.group(1)
            matching_envs = [valid_env_name for valid_env_name, valid_env_spec in self.env_specs.items()
                             if env_name == valid_env_spec._env_name]
            if matching_envs:
                raise error.DeprecatedEnv('Env {} not found (valid versions include {})'.format(id, matching_envs))
            else:
                raise error.UnregisteredEnv('No registered env with id: {}'.format(id))

    def register(self, id, **kwargs):
        if id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)

# Have a global registry
registry = EnvRegistry()

def register(id, **kwargs):
    return registry.register(id, **kwargs)

def make(id, names_and_args=None):
    return registry.make(id, names_and_args=names_and_args)

def spec(id):
    print("calling spec function from registration.py")
    return registry.spec(id)
