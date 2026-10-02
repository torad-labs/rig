# The environment a prebuilt engine is compiled in: the oldest base a prebuilt supports (Ubuntu
# 22.04: glibc 2.35, libstdc++ and libgomp from GCC 11), NVIDIA's CUDA 13.3 toolkit, and the CMake
# the engine is built with everywhere else (22.04 ships 3.22). The base by digest and CMake by
# sha256, so a release is compiled from the same bytes each time.
FROM nvidia/cuda:13.3.0-devel-ubuntu22.04@sha256:e7cb11515d3931675b57eb26c5e70055e702a4d51320c60ebcd899380ee3c7d5
ARG CMAKE_VERSION=3.31.6
ARG CMAKE_SHA256=5a1133ff103c71eb5120e2cc3de922733e7d8a26a98ae716397e8676adb367bf
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git ninja-build \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/cmake.tar.gz \
      "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
 && echo "${CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum -c \
 && tar -C /opt -xzf /tmp/cmake.tar.gz \
 && ln -s "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake \
 && rm /tmp/cmake.tar.gz
