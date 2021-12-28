FROM python:3.9.9-bullseye

COPY requirements.txt requirements-dev.txt /tmp/
RUN python -m pip install \
    -r /tmp/requirements.txt \
    -r /tmp/requirements-dev.txt

WORKDIR /workarea