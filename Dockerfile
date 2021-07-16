FROM ubuntu:20.04
MAINTAINER Hermann Krumrey <hermann@krumreyh.com>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3 python3-pip wget firefox
RUN pip3 install scikit-learn selenium

RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux32.tar.gz &&\
    tar -xvzf geckodriver* &&\
    chmod +x geckodriver &&\
    mv geckodriver /usr/local/bin/

ADD . bot
RUN cd bot && python3 setup.py install

WORKDIR bot
CMD ["multi-betbot", "-v"]
