#!/bin/bash

docker build -t ga-notebook .
docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work ga-notebook