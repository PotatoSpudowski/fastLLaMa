FROM python:3.8.16-bullseye

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN pip install git+https://github.com/PotatoSpudowski/fastLLaMa.git@main

WORKDIR /app

#Change this to your own model
COPY models/ models/
#Change this to your own example
COPY examples/python/* ./

CMD ["python3", "example.py"]
