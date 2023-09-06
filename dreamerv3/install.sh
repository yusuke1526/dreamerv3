#!/bin/sh
set -eu

cd /gym-carla

pip3 install -r requirements.txt
pip3 install wandb moviepy imageio

ln -s /gym-carla/gym_carla /embodied/gym_carla
ln -s /logdir /embodied/logdir

cd /embodied