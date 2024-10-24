# Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration (SUPE) 

This code accompanies the paper [Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](https://arxiv.org/abs/2410.18076). 

The code is built off of [ExPLORe](https://github.com/facebookresearch/ExPLORe). Our diffusion policy code is adapted from [IDQL](https://github.com/philippe-eecs/IDQL). Our VAE pre-training code is adapted from Seohong's implementation of OPAL used in [HILP](https://arxiv.org/abs/2402.15567).

Before setting up the environment, make sure that MuJoCo and the dependencies for mujoco-py are installed (https://github.com/openai/mujoco-py). Then, run the `create_env.sh` script, which will create the conda environment, clone necessary code for running the HILP baseline, and download the pretrained checkpoints.

# Reproducing Experiments in the Paper

## SUPE (Ours)

### Pretraining

Pretrained checkpoints for all environments are downloaded in `create_env.sh`. Below are the commands used to generate the checkpoints. 

#### AntMaze

```
python run_opal.py --env_name=antmaze-large-diverse-v2 --seed=1 --vision=False
```

Replace the env_name with `antmaze-large-diverse-v2-2`, `antmaze-large-diverse-v2-3`, `antmaze-large-diverse-v2-4` to test different goals on AntMaze Large. For AntMaze Medium, use `antmaze-medium-diverse-v2(-#)`, and for Ultra use `antmaze-ultra-diverse-v0(-#)`.

#### Kitchen
```
python run_opal.py --env_name=kitchen-mixed-v0 --seed=1 --vision=False
```

Replace the env_name with `kitchen-partial-v0` and `kitchen-complete-v0` to test the other tasks. 

#### Visual AntMaze
```
python run_opal.py --env_name=antmaze-large-diverse-v2 --seed=1 --vision=True
```

Replace env_name with `antmaze-large-diverse-v2-2`, `antmaze-large-diverse-v2-3`, `antmaze-large-diverse-v2-4` to test other goals. 

### Online Learning

#### AntMaze

```
python train_finetuning_supe.py --config.backup_entropy=False --env_name=antmaze-large-diverse-v2 --config.num_min_qs=1 --offline_relabel_type=min --use_rnd_offline=True --use_rnd_online=True --seed=1
```

#### Kitchen

```
python train_finetuning_supe.py --config.backup_entropy=False --config.num_min_qs=2 --offline_relabel_type=pred --use_rnd_offline=True --use_rnd_online=True --env_name=kitchen-mixed-v0 --seed=1 --config.init_temperature=1.0
```

#### Visual AntMaze

```
python3 train_finetuning_supe_pixels.py --config.backup_entropy=False --config.num_min_qs=2 --config.num_qs=10 --offline_relabel_type=min --use_rnd_offline=True --use_rnd_online=True  --seed=1 --env_name=antmaze-large-diverse-v2 --use_icvf=True
```

## Baseline: Online w/ Trajectory Skills

To run the baseline **Online w/ Trajectory Skills**, use the same commands as above but add `offline_ratio=0` and set `use_rnd_offline=False`. For example, on AntMaze: 

```
python train_finetuning_supe.py --config.backup_entropy=False --env_name=antmaze-large-diverse-v2 --config.num_min_qs=1 --offline_relabel_type=min --use_rnd_offline=False --use_rnd_online=True --seed=1 --offline_ratio=0 
```

## Baselines: HILP w/ Offline Data and Online w/ HILP Skills

The HILP skills were pretrained using the official codebase: https://github.com/seohongpark/HILP, and the pretrained checkpoints can be downloaded using `create_env.sh`. To run the HILP baselines, use the `train_finetuning_supe_hilp.py` and `train_finetuning_supe_pixels_hilp.py` scripts with the same command parameters as **Ours**/**Online w/ Trajectory Skills**. For example, to benchmark on AntMaze, run the following command:

### HILP w/ Offline Data

```
python train_finetuning_supe_hilp.py --config.backup_entropy=False --env_name=antmaze-large-diverse-v2 --config.num_min_qs=1 --offline_relabel_type=min --use_rnd_offline=True --use_rnd_online=True --seed=1
```

### Online w/ HILP Skills

```
python train_finetuning_supe_hilp.py --config.backup_entropy=False --env_name=antmaze-large-diverse-v2 --config.num_min_qs=1 --offline_relabel_type=min --use_rnd_offline=False --use_rnd_online=True --seed=1 --offline_ratio=0 
```

## Baseline: ExPLORe

### AntMaze
```
python train_finetuning_explore.py --config.backup_entropy=False --config.num_min_qs=1 --project_name=explore --offline_relabel_type=min --use_rnd_offline=True --use_rnd_online=True --env_name=antmaze-large-diverse-v2 --seed=1 --rnd_config.coeff=2
```

### Kitchen
```
python train_finetuning_explore.py --config.backup_entropy=False --config.num_min_qs=2 --project_name=explore --offline_relabel_type=pred --use_rnd_offline=True --use_rnd_online=True --env_name=kitchen-mixed-v0 --seed=1 --rnd_config.coeff=2 --config.init_temperature=1.0
```

### Visual AntMaze
```
python train_finetuning_explore_pixels.py --config.backup_entropy=False --config.num_min_qs=1 --config.num_qs=10 --project_name=explore-pixels --offline_relabel_type=min --use_rnd_offline=True --use_rnd_online=True --seed=1 --env_name=antmaze-large-diverse-v2 --updates_per_step=2  --use_icvf=True --rnd_config.coeff=2
```

## Baseline: Online

To run the **Online** baseline, use the same commands as for **ExPLORe** except add `offline_ratio=0` and change `use_rnd_offline=False`. For example, on AntMaze:

```
python train_finetuning_explore.py --config.backup_entropy=False --config.num_min_qs=1 --project_name=explore --offline_relabel_type=min --use_rnd_offline=False --use_rnd_online=True --env_name=antmaze-large-diverse-v2 --seed=1 --rnd_config.coeff=2 --offline_ratio=0 
```

## Baseline: Diffusion BC + JSRL

### AntMaze 

```
python train_finetuning_explore.py --config.backup_entropy=False --config.num_min_qs=1 --project_name=diff_bc_jsrl --offline_relabel_type=min --use_rnd_offline=False --use_rnd_online=True --env_name=antmaze-large-diverse-v2 --seed=1 --rnd_config.coeff=2 --offline_ratio=0 --jsrl_ratio=0.9 --jsrl_discount=0.99 --config.init_temperature=1.0
```

### Kitchen

```
python train_finetuning_explore.py --config.backup_entropy=False --config.num_min_qs=2 --project_name=diff_bc_jsrl --offline_relabel_type=pred --use_rnd_offline=False --use_rnd_online=True --env_name=kitchen-mixed-v0 --seed=1 --rnd_config.coeff=2.0 --config.init_temperature=1.0 --offline_ratio=0 --jsrl_ratio=0.75
```

### Visual AntMaze

```
python train_finetuning_explore_pixels.py --config.backup_entropy=False --config.num_min_qs=1 --config.num_qs=10 --project_name=diff_bc_jsrl_pixels --offline_relabel_type=min --use_rnd_offline=False --use_rnd_online=True --seed=1 --env_name=antmaze-large-diverse-v2 --offline_ratio=0 --updates_per_step=2 --use_icvf=True --rnd_config.coeff=2 --jsrl_ratio=0.9
```

# Bibtex

```
@inproceedings{
wilcoxson2024leveraging,
title={Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration},
author={Max Wilcoxson and Qiyang Li and Kevin Frans and Sergey Levine},
booktitle={Arxiv},
year={2024},
url={https://arxiv.org/abs/2410.18076}
}
```
