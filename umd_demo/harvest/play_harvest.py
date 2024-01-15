# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple human player for playing the `Harvest` level interactively.

Use `WASD` keys to move the character around. `Q` and `E` to turn.
"""
from typing import Any, Callable, Optional, Union

import collections

import dm_env
import dmlab2d
import numpy as np
from absl import app, flags

from meltingpot.human_players import level_playing_utils
from meltingpot.meltingpot.human_players.level_playing_utils import _split_key
from meltingpot.meltingpot.utils.substrates.builder import Settings, builder
from meltingpot.meltingpot.utils.substrates.wrappers.base import Lab2dWrapper
from umd_demo.harvest.configs.environment import harvest_finished as game

FLAGS = flags.FLAGS

flags.DEFINE_integer("screen_width", 800, "Width, in pixels, of the game screen")
flags.DEFINE_integer("screen_height", 600, "Height, in pixels, of the game screen")
flags.DEFINE_integer("frames_per_second", 8, "Frames per second of the game")
flags.DEFINE_string("observation", "RGB", "Name of the observation to render")
flags.DEFINE_bool("verbose", False, "Whether we want verbose output")
flags.DEFINE_bool("display_text", False, "Whether we to display a debug text message")
flags.DEFINE_string(
    "text_message",
    "This page intentionally left blank",
    "Text to display if `display_text` is `True`",
)


_ACTION_MAP = {
    "move": level_playing_utils.get_direction_pressed,
    "turn": level_playing_utils.get_turn_pressed,
}

"""
 player values/types are random between 0 and 1. represents how much they value egalitarianism. 
 1 is most egalitarian, 0 is least egalitarian (utilitarian)
"""
NUM_VOTING_ROUNDS = 1
PLAYER_VALUES = np.random.uniform(size=[game.numPlayers])


def utilitarian(num_apples: dict[str, int]) -> float:
    """Utilitarian objective"""
    return sum(num_apples.values()) / game.numPlayers


def egalitarian(num_apples: dict[str, int]) -> float:
    """Egalitarian objective"""
    return min(num_apples.values())


def vote(player_values: np.ndarray) -> Callable:
    """Vote on objective"""
    if player_values.mean() > 0.5:
        return egalitarian
    else:
        return utilitarian


class Principal:
    """Principal class for universal mechanism design
    In this setting, the principal computes a tax on the reward based on the number of apples collected by the agent.
    The tax should be between 0 and 1.5. 0 means no tax, 1.5 means 150% tax.
    """

    def __init__(self, objective: Callable) -> None:
        self.set_objective(objective)

    def set_objective(self, objective: Callable) -> None:
        print("********\nSetting objective to", objective.__name__, "\n********")
        self.objective = objective

    def __call__(self, num_apples) -> Any:
        """very simple baseline principal: no tax on utilitarian, 100% tax on egalitarian if num_apples > 10"""
        if self.objective.__name__ == "utilitarian":
            return 0
        if self.objective.__name__ == "egalitarian":
            if num_apples > 10:
                return 1.5  # punish the agent for being too greedy
            else:
                return 0


principal = Principal(egalitarian)


class UMDWrapper(Lab2dWrapper):
    """Wrapper that rebuilds the environment on reset."""

    def __init__(self, env: dmlab2d.Environment):
        """Initializes the object.

        Args:
          build_environment: Called to build the underlying environment.
        """
        super().__init__(env)
        self.env = env
        self.apples = collections.defaultdict(int)
        self.collected_tax = 0

    def step(self, action) -> Union[dm_env.transition, dm_env.termination]:
        """Rebuilds the environment and calls reset on it."""
        timestep: dm_env.TimeStep = super().step(action)
        for key in timestep.observation.keys():
            if key.endswith(".REWARD"):
                player_prefix, name = _split_key(key)
                if name != "REWARD":
                    continue
                self.apples[player_prefix] += timestep.observation[key]
                if timestep.observation[key] > 0:
                    if timestep.observation[key] != 1:
                        raise Exception("Reward is not 1")
                    # add tax from principal
                    tax = principal(self.apples[player_prefix])
                    timestep.observation[key] = 1 - tax
                    self.collected_tax += tax

        return timestep


def umd_builder(
    lab2d_settings: Settings,
    prefab_overrides: Optional[Settings] = None,
    env_seed: Optional[int] = None,
    **settings,
) -> dmlab2d.Environment:
    """Universal mechanism design builder"""
    env = builder(lab2d_settings, prefab_overrides, env_seed, **settings)
    return UMDWrapper(env)


def verbose_fn(unused_timestep, unused_player_index: int) -> None:
    pass


def text_display_fn(unused_timestep, unused_player_index: int) -> str:
    return FLAGS.text_message


def main(argv):
    del argv  # Unused.
    for i in range(NUM_VOTING_ROUNDS):
        principal_objective = vote(PLAYER_VALUES)  # vote on objective
        principal.set_objective(principal_objective)
        level_playing_utils.run_episode(
            FLAGS.observation,
            {},  # Settings overrides
            _ACTION_MAP,
            game.get_config(),
            level_playing_utils.RenderType.PYGAME,
            FLAGS.screen_width,
            FLAGS.screen_height,
            FLAGS.frames_per_second,
            verbose_fn if FLAGS.verbose else None,
            text_display_fn if FLAGS.display_text else None,
            env_builder=umd_builder,
        )


if __name__ == "__main__":
    app.run(main)
