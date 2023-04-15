FROM ubuntu:18.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set up the environment to run fastLLaMA

# Add GCC and G++ New
RUN apt-get update && \
    apt-get install software-properties-common curl git -qy && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 42D5A192B819C5DA && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # Add CMAKE New
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ bionic main" && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4 && \
    apt-get update && \
    # Install
    apt-get install -qy cmake gcc-10 g++-10 && \
    # Configure
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 30 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 30 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 && \
    update-alternatives --set cc /usr/bin/gcc && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 && \
    update-alternatives --set c++ /usr/bin/g++ && \
    apt-get install -y python3.9-dev python3.9-distutils python3.9 python3-pip

WORKDIR /app
COPY ./ ./
RUN python3.9 -m pip install --upgrade setuptools pip distlib && \
    python3.9 -m pip install --upgrade -r requirements.txt && \
    python3.9 setup.py -l python

CMD ["python3.9", "example.py"]
