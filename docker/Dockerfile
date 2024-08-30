# Note: CUDA dev version is required to install `profluent` package
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

# Install Mambaforge & put it in the path
# Could just be lazy and use conda instead
# ENV CONDA_DIR /opt/conda
# ENV PATH=$CONDA_DIR/bin:$PATH
# RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh -O ~/Mambaforge-$(uname)-$(uname -m).sh && \
#      /bin/bash ~/Mambaforge-$(uname)-$(uname -m).sh -b -p $CONDA_DIR && \
#      source ~/.bashrc

#RUN conda install -n base -c conda-forge mamba

# Set up `progen-conditional` Mamba environment with pip/Conda dependencies
RUN mkdir progen-conditional
COPY environment.yml progen-conditional/environment.yml
# Note: We use `mamba` here (instead of `conda`) to create the `profluent` environment
# since Mamba's C++ package management backend is often much faster and more accurate for
# dependency resolution compared to `conda`
RUN conda env create -f progen-conditional/environment.yml && conda clean -afy
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# Note: One should still use `conda` to (de)activate Mamba environments,
# but in all other contexts the `mamba` command can safely replace
# the `conda` command (e.g., `mamba install -c conda-forge gcc`)
RUN echo "conda deactivate && conda activate progen-conditional" >> ~/.bashrc

#Install CLEAN environment
# RUN git clone https://github.com/tttianhao/CLEAN.git
# RUN cd CLEAN/app/
# RUN conda create -n clean python==3.10.4 -y
# RUN conda activate clean
# RUN pip install -r requirements.txt
# RUN pip install pytorch
# RUN pip install ipykernel
# RUN python build.py install
# RUN git clone https://github.com/facebookresearch/esm.git
# RUN mkdir data/esm_data
# RUN cd data


SHELL ["conda", "run", "-n", "progen-conditional", "/bin/bash", "-c"]

#SHELL ["mamba", "run", "-n", "progen-conditional", "/bin/bash", "-c"]
