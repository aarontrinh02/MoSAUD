import d4rl
import gym
import numpy as np

from supe.data.dataset import Dataset


class D4RLDataset(Dataset):
    def __init__(
        self,
        env: gym.Env,
        clip_to_eps: bool = True,
        eps: float = 1e-5,
        delete_traj_ends=False,
        subtract_one=True,
        remove_kitchen_goal=False,
    ):
        dataset_dict = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dataset_dict["terminals"][-1] = 1

        dataset_dict["dones"] = np.full_like(dataset_dict["rewards"], False, dtype=bool)
        dataset_dict["traj_ends"] = np.full_like(
            dataset_dict["rewards"], False, dtype=bool
        )

        for i in range(dataset_dict["dones"].shape[0] - 1):
            traj_end = (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
            )
            dataset_dict["traj_ends"][i] = traj_end
            dataset_dict["dones"][i] = traj_end or dataset_dict["terminals"][i] == 1.0

        dataset_dict["dones"][-1] = 1
        dataset_dict["traj_ends"][-1] = 1
        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k in ["observations", "actions", "rewards", "next_observations"]:
            dataset_dict[k] = dataset_dict[k].astype(np.float32)

        for k in ["dones", "masks", "traj_ends"]:
            dataset_dict[k] = dataset_dict[k].astype(bool)

        if subtract_one:
            dataset_dict["rewards"] -= 1.0  # RLPD works better with -1/0 rewards

        if remove_kitchen_goal:
            dataset_dict["observations"] = dataset_dict["observations"][:, :30]
            dataset_dict["next_observations"] = dataset_dict["next_observations"][
                :, :30
            ]

        if delete_traj_ends:
            del dataset_dict["traj_ends"]

        super().__init__(dataset_dict)
