# aps360-fake-news

University of Toronto APS360 project created for the 2020 Summer semester.

## Dependencies

In order to use the model, the required dependencies must be installed.
There are several options for installing dependencies depending on how the model is to be run.
Only one of these installation methods is required.

### via pip

Install `pip` requirements found in [`requirements.txt`](./requirements.txt):

```bash
pip install -r requirements.txt
```

### via Anaconda

Install the `conda` environment found in [`environment.yml`](./environment.yml):

```bash
conda env create -f environment.yml
```

### via Docker

Build the Docker image found in [`Dockerfile`](./Dockerfile):

```bash
docker build -t fake-news .
```

To run the image, use a command such as the following:

```bash
docker run -it --rm fake-news
```

Note that the Docker image comes with the pretrained model already installed.

## Usage

To use the model, simply run [`main.py`](./main.py) and follow the instructions.
For more information, the help menu can be accessed using the `--help` flag:

```bash
./main.py --help
```

### Pretrained Model

To access the pretrained model, [download][pretrained-model] the pretrained model files.
Unzip the archive and place it in the root of this repository.

Alternatively, the pretrained model can be downloaded directly from the command line using the `gdown` utility:

```bash
pip install gdown
gdown https://drive.google.com/uc\?id\=1E7WfQ86vjrwrhIKcEaWKRkB8phpExOLK
tar zxvf output.tar.gz
rm output.tar.gz
```

[pretrained-model]: https://drive.google.com/uc\?id\=1E7WfQ86vjrwrhIKcEaWKRkB8phpExOLK
