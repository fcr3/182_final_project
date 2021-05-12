# Increasing Training Efficiency of Convolutional Neural Policy Networks for Deep Reinforcement Learning

## CS182 Final Project

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

## Files

There are 4 different iterations of Tmpnet inside the `models` folder.

1. `tmpv1.py`: First iteration of template matching. Using the templates as a pre-processing layer on the observation RGB space prior to passing into convolutional policy network. Tested with both fixed templates and trainable templates. Both achieved underwhelming performance.

2. `tmpv2.py`: Second iteration of template matching. Using the templates as a pre-processing layer on the observation RGB space prior to passing into convolutional policy network. However, the input to the convolutional policy network was a concatenation of the output of the template matching and the original RGB space. Performance was good but analysis showed the network was relying entirely on the original RGB space and the template matching channels were disregarded.

3. `tmpv3.py`: Third iteration of template matching. Using the templates as a pre-processing layer, fed through a trainable feed forward network prior to being convolved over the observation RGB space. Achieved underwhelming performance.

4. `tmp_init.py`: Orthonormal template initialization of convolutional kernels.

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

### Evaluating Train Models

Assuming you are in the `rl_project` directory, here is an example command taht you can run to eval a trained model:

```
(final) $ python eval_procgen_tmp.py \
        > <PATH TO TRAINED MODEL (.pt)> \
        > --env-name fruitbot \
        > --distribution-mode easy \
        > --TMPv {choose: v1, v2, v3, init} \
        > --method-label {make: a helpful folder name} \
        > --exp-name {make: a helpful folder name}
        > --num-levels {enter:int} \
        > --start-level {enter: int} \
        > --num-envs {enter: int} \
        > --max_resets {enter: int}
```

If you don't specify a specific `--log-dir` in the terminal, then outputs will default to `logs_tmp/fruitbot/nlev_{num-levels}_easy/{method-label}/{exp-name}`. Furthermore, a folder titled with the datetime within the output folder will hold two folders: `train` and `test`. These folders will hold tf.events corresponding to their specific task (i.e. `train` will have a tf.event holding training statistics).

The eval script will run until at least `max_resets` number of episode terminations.

## Notebooks

The `notebooks` folder within `rl_project` contains notebooks that we used to run the python train and eval scripts on Google Collaboratory. It also contains a few analysis notebooks under the subfolder `analysis_notebooks`. They perform simple plotting of train and eval logs outputted by `train_procgen_tmp.py`.

## Logs

The `logs` folder contains a large number of train and eval logs from our experiments with template initialization. Each directory contains a tfevents file as well as a traditional log, `progress.csv`. One of the test runs from `ortho_template_init_eval` has been moved to the outermost directory and renamed `best_test_log.csv`. The format of the eval logs are rows containing the reward at the end of one episode on an unseen test level. This is submission variant number two. There are also logs showing the testing curve during training inside the training logs, but the logged values are not precisely average episodic reward on unseen test levels. The graphs within our writeup contain these `eval` curves during training, but they range up to a value of 25, while the average episodic reward actually only ranges up to a value of 15. The key for the `eval` curves during training in the training `process.csv`'s is `eval_eprewmean`.
