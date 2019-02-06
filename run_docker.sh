#!/bin/bash

NV_GPU=0 nvidia-docker run -it -p 8888:8888 \
    -v /home/$USER/:/home/$USER mfe7/cadrl_ros:v0
