FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update                              \
    && apt-get install -y --no-install-recommends   \
       python3 git make pipenv cron bzip2 tzdata

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
	&& localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANG=en_US.
ENV PYTHONPATH="/home/bourse/.local/lib/python3/site-packages/"

RUN cp /usr/share/zoneinfo/Europe/Paris /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

RUN adduser --disabled-password --gecos "" bourse
RUN chmod 755 /home/bourse

USER bourse
WORKDIR /home/bourse

COPY Pipfile .
RUN pipenv install

ADD apps.tgz .

CMD pipenv run python3 etl.py
