FROM fedncf:latest

LABEL maintainer="chen.junbao@outlook.com"

ADD . /server

WORKDIR /server

EXPOSE 10000 20000-20004

ENTRYPOINT [ "python", "-u", "main.py" ]