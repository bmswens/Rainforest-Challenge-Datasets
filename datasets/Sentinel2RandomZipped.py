# built in
import argparse
import os
from zipfile import ZipFile
from types import FunctionType

# 3rd party
import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

toTensor = ToTensor()

class Sentinel2RandomZipped(Dataset):
    """
    PyTorch Dataset for the Sentinel-2 Random Split dataset.
    For use on systems where unpacking is not advised.

    Args:
        root (string, optional): Root directory of dataset where ``sentinel2`` exists.
            Defaults to ``data``.
        train (bool, optional): If True, creates dataset from ``{root}/sentinel2-train.zip``,
            otherwise from ``{root}/sentinel2-test.zip```.
            Defaults to ``True``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
            Defaults to ``False```.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
            Defaults to ``None``.
    """

    # file locations
    meta_url = "https://rainforestchallenge.blob.core.windows.net/cvpr/dataset_info.txt"

    train_url = "https://rainforestchallenge.blob.core.windows.net/cvpr/random_split/train/sent2.zip"
    test_url = "https://rainforestchallenge.blob.core.windows.net/cvpr/random_split/test/sent2.zip"

    def __init__(
        self,
        root: str = "data",
        train: bool = True,
        image_mode: str = "L",
        transform: FunctionType = None,
        download: bool = False,
        **kwargs
    ) -> None:
        # labels
        self.train = train
        self.label = "train" if train else "test"

        # paths
        self.root = root
        self.tgt = os.path.join(self.root, f"sentinel2-{self.label}.zip")
        self.tgt_url = self.train_url if train else self.test_url

        # transforms
        self.image_mode = image_mode
        self.transform = transform

        # download if required
        self.download = download
        if download:
            self._download()

        if not self._files_exist():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # get files info
        with ZipFile(self.tgt, 'r') as zipped:
            self.images = [path for path in zipped.namelist() if '.jpg' in path]
            self.size = len(self.images)


    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> dict:
        file_name = self.images[index]
        full_path = os.path.join(self.tgt, file_name)
        with ZipFile(self.tgt, 'r') as zipped:
            with zipped.open(file_name, 'r') as f:
                img = Image.open(f).convert(self.image_mode)
        if self.transform:
            img = self.transform(img)
        tensor = toTensor(img)
        output = {
            "tensor": tensor,
            "image": img,
            "path": full_path
        }
        return output


    def __repr__(self) -> str:
        return f"Sentinel2RandomZipped(root='{self.root}', train={self.train}, transform={self.transform}, download={self.download})"


    def _files_exist(self, print_ok=True):
        return os.path.exists(self.tgt)


    def _download(self):
        # check whether we should download the files and return if already present
        already_exist = self._files_exist()
        if already_exist:
            return
        # make parent folder
        os.makedirs(self.root, exist_ok=True)

        # download zip file to root
        print("Downloading files.")
        response = requests.get(self.tgt_url, stream=True)
        with open(self.tgt, 'wb') as output:
            for chunk in response.iter_content(chunk_size=128):
                output.write(chunk)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The Sentinel 2 Random Split Dataset")
    parser.add_argument(
        "--download", 
        dest="download",
        action="store_true",
        default=False,
        help="Download the dataset (default: False)" 
    )
    parser.add_argument(
        "--test",
        dest="train",
        default=True,
        action="store_false",
        help="Use data set aside for testing (default: False)"
    )
    parser.add_argument(
        '--verify',
        dest="verify",
        default=False,
        action="store_true",
        help="Display the first item to visually verify."
    )
    parser.add_argument(
        "root",
        metavar="path",
        help="The root path to data storage"
    )
    kwargs = vars(parser.parse_args())
    dataset = Sentinel2RandomZipped(**kwargs)
    print(dataset)
    if kwargs["verify"]:
        data = dataset[0]
        print(data)
        data["image"].show()
    print("All good to go, good luck!")
