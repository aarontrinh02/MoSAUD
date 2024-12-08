import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from supe.utils import get_observation_at_index_in_chunk, tft

from .dataset import Dataset

class ChunkDataset(Dataset):
    batch_size: int

    @classmethod
    def create(
        cls,
        dataset,
        chunk_size,
        agent,
        tanh_converter,
        label_skills=False,
        discount=0.99,
        batch_size=4096,
        debug=False,
        skill_dim=8,
        hilp=False,
    ):
        dataset_dict = dataset.dataset_dict

        dataset_dict["modified"] = np.zeros(
            (dataset_dict["actions"].shape[0], 1), dtype=bool
        )
        dataset_dict["skills"] = np.zeros(
            (dataset_dict["actions"].shape[0], skill_dim), dtype=np.float32
        )

        return cls(
            dataset_dict=dataset_dict,
            chunk_size=chunk_size,
            batch_size=batch_size,
            agent=agent,
            discount_rate=discount,
            label_skills=label_skills,
            tanh_converter=tanh_converter,
            debug=debug,
            hilp=hilp,
        )

    def _set_key_at_indices(self, data, indxs, key):
        if type(data) == dict or type(data) == FrozenDict:
            for subkey in data.keys():
                self.dataset_dict[key][subkey][indxs] = data[subkey]
        else:
            self.dataset_dict[key][indxs] = data

    def update_dataset_dict(self, batch, indxs):
        for key in batch.keys():
            self._set_key_at_indices(batch[key], indxs, key)

    def __init__(
        self,
        dataset_dict,
        chunk_size,
        batch_size,
        agent,
        discount_rate,
        label_skills,
        tanh_converter,
        debug=False,
        hilp=False,
    ):
        self._chunk_size = chunk_size
        self._allowed_indx = []
        self.batch_size = batch_size
        self.discount_rate = discount_rate
        self.label_skills = label_skills
        self.hilp = hilp
        for i in range(dataset_dict["traj_ends"].shape[0]):
            if (
                i + chunk_size - 1 < dataset_dict["traj_ends"].shape[0]
                and dataset_dict["traj_ends"][i : i + chunk_size - 1].sum() == 0
            ):
                self._allowed_indx.append(i)
        del dataset_dict["traj_ends"]
        super().__init__(dataset_dict)

        self._allowed_indx = np.array(self._allowed_indx)
        if self.label_skills:
            index = 0
            if debug:
                self._allowed_indx = self._allowed_indx[: self.batch_size]

            while index * self.batch_size < len(self._allowed_indx):
                indx = self._allowed_indx[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]
                chunk_indx = indx[..., None] + np.arange(self._chunk_size)
                batch = (
                    super()
                    .sample(self.batch_size * self._chunk_size, indx=chunk_indx)
                    .unfreeze()
                )
                assert (
                    batch["modified"].sum() == 0
                )  # sanity check to make sure we don't sample something we have already processed

                if self.hilp:
                    curr_phi = agent.get_phi(
                        tft(
                            get_observation_at_index_in_chunk(
                                batch["observations"], index=0
                            )
                        )
                    )
                    next_phi = agent.get_phi(
                        tft(
                            get_observation_at_index_in_chunk(
                                batch["next_observations"], index=-1
                            )
                        )
                    )
                    skills = next_phi - curr_phi
                    skills = skills / np.linalg.norm(
                        skills, axis=-1, keepdims=True
                    )  # normalize
                else:
                    skills = agent.vae(
                        agent.train_state.params,
                        batch["observations"],
                        batch["actions"],
                        method="encode",
                    )

                batch["observations"] = get_observation_at_index_in_chunk(
                    batch["observations"], index=0
                )
                batch["next_observations"] = get_observation_at_index_in_chunk(
                    batch["next_observations"], index=-1
                )

                skills = tanh_converter.to_tanh(skills)
                batch["skills"] = skills
                del batch["actions"]

                reward_discounts = np.power(
                    self.discount_rate, np.arange(self._chunk_size)
                )
                seq_rewards = batch["rewards"] * reward_discounts
                masks = 1 - (np.cumsum(batch["masks"] == 0, axis=-1) > 0)
                seq_rewards = jnp.concatenate(
                    [seq_rewards[:, [0]], seq_rewards[:, 1:] * masks[:, :-1]], axis=-1
                )
                batch["rewards"] = seq_rewards.sum(axis=-1)
                batch["masks"] = np.min(masks, axis=1)

                dones = np.cumsum(batch["dones"] == 1, axis=-1) > 0
                batch["dones"] = np.max(dones, axis=1)
                batch["modified"] = np.ones((dones.shape[0], 1), dtype=bool)

                self.update_dataset_dict(batch, indx)
                index += 1
            self.dataset_dict["actions"] = self.dataset_dict["skills"]
            del self.dataset_dict["skills"]

    def sample_chunk(self, sample_shape, rng):
        indx = np.random.choice(self._allowed_indx, size=sample_shape, replace=False)
        chunk_indx = indx[..., None] + np.arange(self._chunk_size)
        batch = super().sample(self.batch_size * self._chunk_size, indx=chunk_indx)

        return dict(
            seq_observations=batch["observations"],
            next_seq_observations=batch["next_observations"],
            seq_actions=batch["actions"],
            seq_rewards=batch["rewards"],
            seq_masks=1 - (np.cumsum(batch["masks"] == 0, axis=-1) > 0),
        )

    def dataset_dict(self):
        return self.dataset_dict

    def sample(self, sample_shape, indx=None):
        if indx is None:
            if sample_shape is None:
                indx = self._allowed_indx
            else:
                indx = np.random.choice(
                    self._allowed_indx, size=sample_shape, replace=False
                )

        batch = super().sample(self.batch_size, indx=indx).unfreeze()

        del batch["modified"]

        return FrozenDict(batch)
