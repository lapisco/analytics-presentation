docker-compose up --build -d
sleep 10
workon ap
cd presentation && python main.py && cd ..