sudo docker compose up --build -d
sleep 50
source /usr/local/bin/virtualenvwrapper.sh
workon feira
cd presentation && python main.py && cd ..
sudo docker-compose down