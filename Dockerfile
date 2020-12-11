FROM ubuntu:20.04
MAINTAINER Hermann Krumrey <hermann@krumreyh.com>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3 python3-pip qt5-default libqt5webkit5-dev xvfb python3-lxml
RUN pip3 install tensorflow keras dryscrape

ADD . bot
RUN cd bot && python3 setup.py install
RUN mkdir -p /root/.config/betbot && cp -r bot/models/* /root/.config/betbot

WORKDIR bot
CMD ["docker_start.sh", "-v", "--loop"]
