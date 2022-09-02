# SAR_RARP50 toolkit

The repository provides data manipulation code for the SAR-RARP50
dataset, introduced as a part of the EndoVis MICCAI 2022 challenge. The dataset
provides action recognition and surgical instrumentation segmentation labels
reference for 50 RARP procedure clips, focusing on the DVC suturing phase of the
procedure. For more information about the dataset and the challenge please visit
the [page of the challenge](https://www.synapse.org/#!Synapse:syn27618412/wiki/)

## Setup

To run the SAR-RARP50 code you need to either pull a docker image from dockerhub.

### Pulling a docker image

We recommend using a docker container to run the SAR-RARP50 toolkit code.
Currently, we only support building this docker image from source or pulling
our docker image from dockerhub:

Clone this repository and cd into it

```bash
git clone https://github.com/Project-zero-one/SAR_RARP50-evaluation && cd ./SAR_RARP50-evaluation
```

Build the docker image from source.

```bash
docker image build -t yumion7488/sar-rarp50 .
```

Pull the docker image from dockerhub

```bash
docker pull yumion7488/sar-rarp50:latest
```

## How to use

For the SAR-RARP50 Endovis challenge we provide code to generate
predictions in the format we expect competing methods to export results.

To use any of the provided scripts, you must mount the SAR-RARP50 dataset directory
as a volume in the docker container. By doing so, predictions can be
stored under the SAR-RARP50 dataset directory.

For each command, you will need to run the docker container as follows:

``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data \
                     yumion7488/sar-rarp50:latest \
                     args
```

The `-v` flag mounts the directory containing a local copy of the SAR-RARP50 dataset
to the /data directory inside the docker container. Be sure to provide the absolute
path of your local data directory. The SAR-RARP50 dataset directory is assumed to have
the following file structure

```tree
path_to_root_data_dir
├── train1
│   ├── video_*
│   ├── ...
│   └── video_*
├── train2
│   ├── video_*
│   ├── ...
│   └── video_*
└── test
    ├── video_*
    ├── ...
    └── video_*

```

Replace args according to the task you want to perform

### Generating predictions

To generate predictions run the following

``` bash
docker container run --rm \
                     -v /path_to_root_data_dir/:/data/ \
                     yumion7488/sar-rarp50:latest \
                     path/to/test/dir path/to/prediction/dir segmentation
```

The script exposes the following command line interface

test_dir prediction_dir task [--overwrite]

- `test_dir` : Absolute path of the test directory inside the container
- `prediction_dir` : Absolute path of the directory to store the mock predictions under
- `task` : segmentation or actions or multitask
- [`-o`, `--overwrite`] : Flag to overwrite the mock predictions if prediction_dir exists.
