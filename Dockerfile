FROM ubuntu:22.04

ENV SUMO_HOME /opt/sumo

RUN apt-get update
RUN apt-get install -y sumo sumo-tools sumo-doc python3-pip

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "src/main.py"]
