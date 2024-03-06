#!/bin/sh

sudo docker stop $(docker ps -aq)
sudo docker rm $(docker ps -aq)
sudo docker volume rm $(sudo docker volume ls -q)
sudo docker system prune --all --force --volumes