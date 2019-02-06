FROM nvidia/cuda
LABEL maintainer "Michael Everett <mfe@mit.edu>"

RUN apt update
RUN apt -y upgrade

########################################
# Install ROS
# setup keys
RUN apt-key adv --keyserver ha.pool.sks-keyservers.net --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO kinetic
RUN apt-get update && apt-get install -y \
    ros-kinetic-ros-core \
    && rm -rf /var/lib/apt/lists/*


########################################

########################################
# Install tensorflow-gpu w/ python2.7 
RUN apt update
RUN apt-get -y install python2.7 python-pip python-dev
RUN pip2 install tensorflow
########################################
# Install helpful libraries
#RUN pip3 install keras \
#    scikit-learn 

# collision avoidance domain visualization
RUN apt -y update
RUN apt -y install xvfb
RUN pip install pyyaml rospkg catkin_pkg matplotlib shapely


# Import user environment variable
ARG user
ENV USER $user

# Install Jupyter notebook
RUN pip install jupyter

##########################
# # Set up SSH keys for bitbucket
# RUN mkdir /root/.ssh/
# # Copy over private key, and set permissions
# ADD ./id_rsa /root/.ssh/id_rsa
# # Create known_hosts
# RUN touch /root/.ssh/known_hosts
# # Add bitbuckets key
# RUN ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts
######################################

COPY ./ros_and_openai_entrypoint.sh /

# Make ports available to the outside world
# Jupyter
EXPOSE 8888
# TensorBoard
EXPOSE 6006

ENTRYPOINT ["/ros_and_openai_entrypoint.sh"]
CMD ["bash"]

# As a reminder, this command will run the collision avoidance code (w/ GUI forwarding???) within the docker img
# xvfb-run -s "-screen 0 1400x900x24" python code_to_run.py
