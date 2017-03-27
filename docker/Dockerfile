FROM nvidia/cuda:7.5-cudnn3-devel-ubuntu14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libmatio-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-opencv \
        python-pip \
        python-setuptools \
        python-scipy && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 https://bitbucket.org/deeplab/deeplab-public.git && \
    cd deeplab-public && mkdir build && cd build && \
    cmake -DBUILD_EXAMPLES=OFF -DCMAKE_CXX_STANDARD_LIBRARIES="-lmatio" .. && \
    make -j"$(nproc)" && cd ../.. && \
    git clone --depth 1 https://github.com/suyogduttjain/pixelobjectness.git
    
RUN pip install --upgrade pip && \
    pip install flask && \
    pip install flask_cors && \
    pip install pillow
