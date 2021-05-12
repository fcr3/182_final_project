# CS 182 Final Project

This repository serves as the codebase for our CS 182 Final Project.

## Instructions

1. It is recommended to start an environment first using virtual env or conda:
```
$ conda create --name final python=3.7
```

2. Install the necessary libraries:
```
$ conda activate final
(final) $ cd rl_project
(final) $ pip install -r requirements.txt
```

Optionally, CUDA can be installed in order to hasten training. This will provide an enhanced experience, but is not necessary.

## Running the Demos

Our project tested a number of different ways to integrate Template Matching into a deep reinforcement learning system. To run these demos, follow the instructions below.

### Running the TMPNet Demos

Assuming you are in the `rl_project` directory, here is an example command that you can run to train and eval the model along the way:

```
(final) $ python train_procgen_tmp.py \
        > --env-name fruitbot \
        > --distribution-mode easy \
        > --TMPv {choose: v1, v2, v3, init} \
        > --method-label {make: a helpful folder name} \
        > --exp-name {make: a helpful folder name}
        > --num-levels {enter:int} \
        > --start-level {enter: int} \
        > --num-envs {enter: int} \
        > --max-steps 5_000_000
```

If you don't specify a specific `--log-dir` in the terminal, then outputs will default to `logs_tmp/fruitbot/nlev_{num-levels}_easy/{method-label}/{exp-name}`. Furthermore, a folder titled with the datetime within the output folder will hold two folders: `train` and `test`. These folders will hold tf.events corresponding to their specific task (i.e. `train` will have a tf.event holding training statistics).
