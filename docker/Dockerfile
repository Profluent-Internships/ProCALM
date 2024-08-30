FROM --platform=linux/x86_64 pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

SHELL ["/bin/bash", "-c"]
WORKDIR /root

# Install base utilities
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        vim \
        libxml2 \
        apt-transport-https \
        ca-certificates \
        gnupg && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir progen-conditional
COPY environment.yml progen-conditional/environment.yml
RUN conda env create -f progen-conditional/environment.yml && conda clean -afy
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda deactivate && conda activate progen-conditional" >> ~/.bashrc

SHELL ["conda", "run", "-n", "progen-conditional", "/bin/bash", "-c"]
