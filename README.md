# Analytics Presentation

## Stream

Stream module reads webcam image and saves a frame in `stream` folder. This frame is used by all analytics modules.


## Presentation

Presentation module reads processed images from selected analytics and build a presentation with this images and a specified background. Choose a background image and save with name `overlay.png`.

## Run project

To run this project is used Docker Compose, Nvidia Container Toolkit and CUDA.

### Build and Up containers

Run selected containers and Stream module.

```sh
docker-compose up --build -d
```

Install presentation requirements.

```sh
cd presentation
pip install -r requirements.txt
```

Run Presentation module.

```sh
python main.py
```

