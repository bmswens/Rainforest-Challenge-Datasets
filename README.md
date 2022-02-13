# Rainforest Challenge Datasets
This repository contains datasets for working on the [Rainforest Challenge](https://rainforestchallenge.blob.core.windows.net/cvpr/dataset_info.txt).

## Installing
```bash
git clone https://github.com/bmswens/Rainforest-Challenge-Datasets.git
cd Rainforest-Challenge-Datasets
```

## Requirements
Python requirements are documented in [requirements.txt](requirements.txt).

Virtual environments are recommended for install dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage
### Command Line Interface (CLI)
The Python files can be invoked directly from the command line in order to verify installation, and/or download the data if needed.

Files can be downloaded and output of dataset can be checked with:
```bash
python3 datasets/Sentinel2Random.py --download --verify data
```
Check an already downloaded dataset with:
```bash
python3 datasets/Sentinel2Random.py --test --verify data
```
CLI Help:
```bash
usage: Sentinel2Random.py [-h] [--download] [--test] [--verify] path

The Sentinel 2 Random Split Dataset

positional arguments:
  path        The root path to data storage

optional arguments:
  -h, --help  show this help message and exit
  --download  Download the dataset (default: False)
  --test      Use data set aside for testing (default: False)
  --verify    Display the first item to visually verify.
```

### Python files
Note: the output of `Dataset.__getitem__()` is  of the following form:
```Python
{
    "tensor": torch.Tensor,
    "image": PIL.Image.Image,
    "path": str
}
```
Usage in a Python file:
```Python
from datasets.Sentinel2Random import Sentinel2Random

dataset = Sentinel2Random('data', download=True)
```

## Notes
Completed:
- Sentinel2Random
- Sentinel2RandomZipped

Planned:
- Sentinel1Random
- Sentinel1Geo
- Sentinel2Geo
- Landsat5Random
- Landsat5Geo
- Landsat8Random
- Landsat8Geo

---
**Normal vs `Zipped` datasets**

The `Sentinel2RandomZipped` dataset has been provided as a proof-of-concept for filesystems that are limited in storage.

This implementation of the `Dataset` class saves storage space on the device, at the cost of drastically longer access time per file.

---

## Contributors
- [Brandon Swenson](https://github.com/bmswens) - Author