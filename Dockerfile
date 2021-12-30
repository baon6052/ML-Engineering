FROM python:3.9.9-bullseye
FROM jupyter/minimal-notebook

COPY requirements.txt requirements-dev.txt /tmp/
RUN python -m pip install \
    -r /tmp/requirements.txt \
    -r /tmp/requirements-dev.txt \
    && jupyter nbextension enable --py widgetsnbextension --sys-prefix \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager

WORKDIR /workarea