FROM ubuntu:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN flatpak install flathub org.eclipse.sumo

COPY . .
