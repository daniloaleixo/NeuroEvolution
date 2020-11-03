FROM jupyter/scipy-notebook:ubuntu-18.04
ENV GRANT_SUDO yes
USER root

RUN pip install --upgrade pip

RUN pip install keras==2.2.4
RUN pip install tensorflow==1.15
RUN pip install gym && pip install keras_metrics && pip install opencv-python

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]