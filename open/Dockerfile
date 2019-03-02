FROM nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

RUN conda install faiss-cpu -c pytorch
RUN conda install scipy
RUN pip install tqdm six flask flask_cors tornado h5py
RUN conda install -c cyclus java-jdk
RUN apt-get update
RUN apt-get install -y git curl
RUN git clone https://github.com/facebookresearch/DrQA.git
RUN cd DrQA; pip install -r requirements.txt; python setup.py develop
# RUN ./install_corenlp.sh