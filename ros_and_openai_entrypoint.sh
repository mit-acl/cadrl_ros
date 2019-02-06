#!/bin/bash
set -e

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash"

cd /home/swarm/Desktop/acl/gym_coll_avoidance_mfe/cadrl_ros/
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root

exec "$@"

