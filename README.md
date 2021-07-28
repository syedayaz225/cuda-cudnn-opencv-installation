# cuda-cudnn-opencv-installation

Installation sequence 
1-cuda
2 cudnn
3-opencv

#####################################################
#1#Installation CUDA
#####################################################
i-base image
ii-runtime
iii-devel

#If using any enviromental variable 
cd
conda deactivate


#*************i) Base image *************

apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

apt-get update
apt-get upgrade


export CUDA_VERSION=11.2.0

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-2=11.2.72-1 \
    cuda-compat-11-2 \
    && ln -s cuda-11.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_REQUIRE_CUDA="cuda>=11.2 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 driver>=450,driver<451"

#*************ii) Runtime *************
export NCCL_VERSION=2.8.4

apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-2=11.2.0-1 \
    libnpp-11-2=11.2.1.68-1 \
    cuda-nvtx-11-2=11.2.67-1 \
    libcublas-11-2=11.3.1.68-1 \
    libcusparse-11-2=11.3.1.68-1 \
    libnccl2=$NCCL_VERSION-1+cuda11.2 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
apt-mark hold libcublas-11-2 libnccl2

#*************iii) development *************

export NCCL_VERSION=2.8.4

#change if this command give error ( in development libnccl-dev=2.8.4-1+cuda11.2 change the required version of libncc-dev )

apt-get update && apt-get install -y --no-install-recommends \
    libtinfo5 libncursesw5 \
    cuda-cudart-dev-11-2=11.2.72-1 \
    cuda-command-line-tools-11-2=11.2.0-1 \
    cuda-minimal-build-11-2=11.2.0-1 \
    cuda-libraries-dev-11-2=11.2.0-1 \
    cuda-nvml-dev-11-2=11.2.67-1 \
    libnpp-dev-11-2=11.2.1.68-1 \
    libnccl-dev=2.8.3-1+cuda11.2 \
    libcublas-dev-11-2=11.3.1.68-1 \
    libcusparse-dev-11-2=11.3.1.68-1 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
apt-mark hold libcublas-dev-11-2 libnccl-dev

export LIBRARY_PATH=/usr/local/cuda/lib64/stubs

nvcc --version

git clone https://github.com/AlexeyAB/darknet.git
cd darknet
nano Makefile
change GPU=1 and save
make

#without error run below command
./darknet test

#Get out of darknet.
cd ..

#####################################################
#2#download cudnn of cuda 11.2(because we installed it)
#####################################################
There are two methods:

1#########go to nvdia and download cudnn version according to your NVidea account.
https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz?JdRS4z9D0IxGzxHsb5YKZs4vXigs1ijC6BytoN6ilGrPl9YuDOk0xVMCx32Wfu4hbATyTZmL9l5InJpJZ7nYsMH8Xb9lCqKVEcDC6R7TNvVnaI1j8b-DnKuS516olXaMhg2Fmys_c41EZkMw3-teI1W72NyEv7GVBK0HniSST05s1JDsF3O9b9gjfDsqnFAgGosuaBi4i9JZDaAnhl4

I downloaded and extracted into my laptop just run the command to copy from my laptop to server machine

scp -P 10607 -r /home/abc/PycharmProjects/DroneNet/dependencies\ cudnn\ for\ uni_server/cudnn-11.2-linux-x64-v8.1.1.33/  root@joffrey.dimis.fim.uni-passau.de:/root/cudnn-11.2-linux-x64-v8/

2############Another method but not much helpful

wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-linux-x64-v8.1.1.33.tgz?JdRS4z9D0IxGzxHsb5YKZs4vXigs1ijC6BytoN6ilGrPl9YuDOk0xVMCx32Wfu4hbATyTZmL9l5InJpJZ7nYsMH8Xb9lCqKVEcDC6R7TNvVnaI1j8b-DnKuS516olXaMhg2Fmys_c41EZkMw3-teI1W72NyEv7GVBK0HniSST05s1JDsF3O9b9gjfDsqnFAgGosuaBi4i9JZDaAnhl4

#rename tgz file to some small file name i.e cudnn-11.2-linux-x64-v8.1.0.77.tgz
mv cudnn-11.2-linux-x64-v8.1.0.77.tgz?R_8Pe-VJ8hZTwUJ5vanAKQcnhJeZYQnpazCYRKBrfRBRIWKuW8AXrSWqGJTAM6uh--IuQ807Jxjkv6v9KIcPgmK7rW_vrrqXP8_r6A536Sj4ezKMUnhvkEn0GdbnomHhVYmflhJbrpPw0TWJZ2r22tMVgWP6Ghqkf7DBs8Brm9BhWcxnBQz57eDl-zdq3tkqhfwYKoFKLdSfNg6uJwo cudnn-11.2-linux-x64-v8.1.0.77.tgz



#unzip
tar zxvf cudnn-11.2-linux-x64-v8.1.0.77.tgz

#install cudnn ubuntu
#go to cuda dir 
cd cuda 

#coping all .h files
cp -P include/*.h /usr/include
cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*

#check cudnn
nvcc --version

#####################################################
#3#Installation OpenCV
#####################################################
apt update
apt install libopencv-dev python3-opencv



#test
cd darknet
nano Makefile
change CUDNN=1 
change CUDNN_HALF=1 
change OPENCV=1 and save
make

#without error run below command
./darknet test
