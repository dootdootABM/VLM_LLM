FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update

WORKDIR /ros2_ws
COPY src ./src

# Filter hardware-only requirements AND inject numpy<2 into the same file
RUN set -eux; \
    REQ_IN="src/drowsiness_detection_pkg/drowsiness_detection/requirements.txt"; \
    python3 -m pip install --no-cache-dir "pip<25"; \
    python3 -m pip install --no-cache-dir "setuptools<58"; \
    grep -vE '^(RPi\.GPIO|evdev)$' "$REQ_IN" > /tmp/requirements.filtered.txt; \
    echo "numpy<2" >> /tmp/requirements.filtered.txt; \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.filtered.txt

RUN apt-get update && rosdep install --from-paths src --ignore-src -r -y
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

COPY docker_test/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
