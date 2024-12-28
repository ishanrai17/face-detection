# docker run -it --volume "$(pwd)":/usr/src/cat-classifier cat-classifier
# python3 -m venv myenv
# source myenv/bin/activate

# Use the latest Ubuntu image as the base
FROM ubuntu:latest

# Set environment variables to prevent prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install Python, pip, and venv
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv libgl1 libglib2.0-0 && \
    apt-get clean

# assign working directory
WORKDIR /usr/src/face-detection

# Set Python3 as the default Python
# RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the default command to launch a shell
CMD ["bash"]
