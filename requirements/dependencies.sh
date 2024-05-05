#!/usr/bin/env bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh || exit 1
/bin/bash /opt/miniconda.sh -b -p /opt/conda || exit 2
. /opt/conda/bin/activate
/opt/conda/bin/conda create -y --name $ENV_NAME --file /requirements/environment.yml || exit 3
/opt/conda/bin/conda init
echo "/opt/conda/bin/conda init" >> ~/.bashrc
# and we remove needless files
rm -rf /opt/miniconda.sh
rm -rf /var/lib/apt/lists/*

apt-get update && apt-get install -y --no-install-recommends vim wget software-properties-common && rm -rf /var/lib/apt/lists/* || exit 4