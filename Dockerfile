FROM nvcr.io/nvidia/pytorch:19.05-py3
FROM php:7.1.9-apache
FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get -y install nano wget curl

# ONNX Runtime Training Module for PyTorch
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
ARG TORCH_CUDA_VERSION=cu111
ARG TORCH_VERSION=1.8.1
ARG TORCHVISION_VERSION=0.9.1

# Install and update tools to minimize security vulnerabilities
RUN apt-get update
RUN apt-get install -y software-properties-common wget apt-utils patchelf git libprotobuf-dev protobuf-compiler cmake
# RUN unattended-upgrade
RUN apt-get autoremove -y

# Python and pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN apt-get install -y python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN pip install --upgrade pip

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip
RUN pip install opencv-python
RUN apt-get install libgl1-mesa-dev -y
RUN apt-get install libegl1-mesa-dev -y

# PyTorch
RUN pip install torch==${TORCH_VERSION}+${TORCH_CUDA_VERSION} torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_VERSION} -f https://download.pytorch.org/whl/torch_stable.html


RUN apt-get -y install vim
ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build