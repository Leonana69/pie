# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images_ros2/create_ros_image.Dockerfile.em
FROM ros:humble-ros-core-jammy

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# setup colcon mixin and metadata
RUN colcon mixin add default \
      https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && \
    colcon mixin update && \
    colcon metadata add default \
      https://raw.githubusercontent.com/colcon/colcon-metadata-repository/master/index.yaml && \
    colcon metadata update

# install ros2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base=0.10.0-1* \
    && rm -rf /var/lib/apt/lists/*

# install gstreamer
RUN apt-get update && apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav

# install slam_toolbox
RUN apt-get update && apt-get install -y ros-humble-slam-toolbox

# install nav2
RUN apt-get update && apt-get install -y ros-humble-navigation2 ros-humble-nav2-bringup

# source the setup files
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /root/.bashrc

COPY ../src /workspace/src
WORKDIR /workspace

RUN /bin/bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && cd /workspace && colcon build"
RUN chmod +x /workspace/src/typego_sdk/scripts/*.sh
CMD ["bash", "-c", "/workspace/src/typego_sdk/scripts/run.sh"]