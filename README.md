# Accelerating Model-Based Reinforcement Learning with Skill Abstraction from Unlabeled Data
 

The code is built off of [SUPE](https://github.com/rail-berkeley/supe).

Before setting up the environment, make sure that MuJoCo and the dependencies for mujoco-py are installed (https://github.com/openai/mujoco-py). Then, run the `create_env.sh` script, which will create the conda environment and download the pretrained checkpoints.

# Reproducing Experiments in the Paper

### Pretraining

Pretrained checkpoints for all environments are downloaded in `create_env.sh`. Below are the commands used to generate the checkpoints. 

#### Kitchen
```
python run_opal.py --env_name=kitchen-mixed-v0 --seed=1 --vision=False
```

Replace the env_name with `kitchen-partial-v0` and `kitchen-complete-v0` to test the other tasks. 

### Online Learning

#### Kitchen

```
python train_finetuning_supe.py --config.backup_entropy=False --config.num_min_qs=2 --offline_relabel_type=pred --use_rnd_offline=True --use_rnd_online=True --env_name=kitchen-mixed-v0 --seed=1 --config.init_temperature=1.0
```
