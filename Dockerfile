FROM ubuntu:20.04
MAINTAINER Hermann Krumrey <hermann@krumreyh.com>

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y python3 python3-pip

ADD . bot
RUN cd bot && python3 setup.py install

WORKDIR bot
CMD ["/usr/bin/python3", "bin/betbot"]
