import argparse as ap
from itertools import chain
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
from nptyping import NDArray


class Config(dict):
    def __init__(self, path: str):
        f = open(path, "r")
        self.__path = path
        d: dict = yaml.safe_load(f)
        [self.__setitem__(item[0], item[1]) for item in d.items()]
        f.close()

    @property
    def narms(self) -> int:
        return self["number-of-arms"]

    @property
    def nlearner(self) -> int:
        return self["number-of-learners"]

    @property
    def weight(self) -> float:
        return self["weight"]

    @property
    def initpreds(self) -> Tuple[float, float]:
        return tuple(self["range-initial-prediction"])

    @property
    def rangelr(self) -> Tuple[float, float]:
        return tuple(self["range-learning-rate"])

    @property
    def reward_probs(self) -> List[List[float]]:
        return self["reward-probabilities"]

    @property
    def trial(self) -> List[int]:
        return self["number-of-trial"]

    @property
    def filename(self) -> str:
        return self["filename"]


class ConfigClap(object):
    def __init__(self):
        self._parser = ap.ArgumentParser()
        self._parser.add_argument("--yaml",
                                  "-y",
                                  help="path to configuration file (`yaml`)",
                                  type=str)
        self._args = self._parser.parse_args()

    def config(self) -> Config:
        yml = self._args.yaml
        return Config(yml)


def softmax(x: NDArray[1, float]) -> NDArray[1, float]:
    x_ = np.exp(x)
    return x_ / np.sum(x_)


class DisAgreeAgent(object):
    def __init__(self, narms: int, num_lerner: int, initpreds: Tuple[float,
                                                                     float],
                 rangelr: Tuple[float, float], weight: float):
        self._k = narms
        self._n = num_lerner
        self._w = weight
        self._preds = np.random.uniform(*initpreds, (narms, num_lerner))
        self._lrs = np.random.uniform(*rangelr, (narms, num_lerner))

    def update(self, rewards: NDArray[1, float], action: NDArray[1, int]):
        tds = rewards.reshape(self._k, -1) - self._preds
        self._preds += self._lrs * tds * action.reshape(self._k, -1)

    def ensemble_preds(self) -> NDArray[1, float]:
        return np.mean(self._preds, axis=1)

    def disagreements(self) -> NDArray[1, float]:
        return self._w * np.var(self._preds, axis=1)

    def choose_action(self, preds: NDArray[1, float],
                      curiosities: NDArray[1, float]) -> NDArray[1, int]:
        val = preds + curiosities
        probs = softmax(val)
        action = np.random.choice(self._k, p=probs)
        return np.identity(self._k)[action]


class NArmBandit(object):
    def __init__(self, probs: NDArray[1, float]):
        self._probs = probs

    def rewards(self):
        return np.random.binomial(1, self._probs)


def to_drow(probs: List[float], preds: NDArray[1, float],
            disagreements: NDArray[1, float], reward: List[float]) -> Tuple:
    return tuple(chain(probs, preds, disagreements, reward))


def colnames(narms: int) -> List[str]:
    probs = [f"prob_{i}" for i in range(narms)]
    preds = [f"pred_{i}" for i in range(narms)]
    disagreements = [f"disagreement_{i}" for i in range(narms)]
    return list(chain(probs, preds, disagreements, ["reward"]))


def get_current_dir(relpath: str) -> Path:
    return Path(relpath).absolute()


def create_data_dir(relpath: str, parent: str):
    cur_dir = get_current_dir(relpath)
    target_dir = cur_dir
    while not target_dir.stem == parent:
        target_dir = target_dir.parent
    data_dir = target_dir.joinpath("data")
    if not data_dir.exists():
        data_dir.mkdir()
    return data_dir


if __name__ == '__main__':
    from pandas import DataFrame
    config = ConfigClap().config()

    # parameters for agent
    rangelr = config.rangelr
    initpreds = config.initpreds
    weight = config.weight

    # parameters for environment
    narms = config.narms
    numlearner = config.nlearner
    trials = config.trial
    probs = config.reward_probs

    agent = DisAgreeAgent(narms, numlearner, initpreds, rangelr, weight)

    results: List[Tuple] = []

    for trial, prob in zip(trials, probs):
        print(f"reward probabilities: {prob}")
        env = NArmBandit(np.array(prob))
        for t in range(1, trial + 1):
            preds = agent.ensemble_preds()
            disagreements = agent.disagreements()
            action = agent.choose_action(preds, disagreements)
            rewards = env.rewards()
            agent.update(rewards, action)
            if t % 10 == 0:
                print(f"{t} - preds: {preds} - curiosity: {disagreements}")
            reward = np.max(action * rewards)
            results.append(to_drow(prob, preds, disagreements, [reward]))

    output_data = DataFrame(results, columns=colnames(narms))
    data_dir = create_data_dir(__file__, "disagree")
    filepath = data_dir.joinpath(config.filename)
    output_data.to_csv(filepath, index=False)
