# We start from the official Python image
ARG PYTHON_VER="3.10"
FROM --platform=linux/x86_64 python:$PYTHON_VER

# Environment variables
ARG ENV_NAME="ConformalPrediction"
ENV ENV_NAME=$ENV_NAME

# Copy files
RUN mkdir /requirements
COPY requirements/* /requirements/

# Install dependencies
RUN /bin/bash /requirements/dependencies.sh $ENV_NAME
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
