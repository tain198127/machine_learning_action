FROM python:3.7
RUN mkdir -p ~/.pip
RUN echo "[global]\n\
trusted-host =  mirrors.aliyun.com\n\
index-url = http://mirrors.aliyun.com/pypi/simple" > ~/.pip/pip.conf
WORKDIR /app

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

COPY ./ ./

ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

ENTRYPOINT ["python3", "/app/__main__.py"]

