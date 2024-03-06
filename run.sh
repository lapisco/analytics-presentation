sudo docker compose up --build -d
sleep 10
source /usr/local/bin/virtualenvwrapper.sh
workon ap
cd presentation && python main.py && cd ..
sudo docker-compose down