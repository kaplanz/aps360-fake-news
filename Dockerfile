FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN pip install gdown && \
    gdown https://drive.google.com/uc\?id\=1E7WfQ86vjrwrhIKcEaWKRkB8phpExOLK && \
    tar zxvf output.tar.gz && \
    rm output.tar.gz

CMD ["bash"]
