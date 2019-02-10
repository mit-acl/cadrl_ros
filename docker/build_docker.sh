#!/bin/bash

docker build --build-arg user=$USER -t mfe7/cadrl_ros:v0 . 
