# built in
import argparse
import os
from zipfile import ZipFile
from types import FunctionType
import shutil

# 3rd party
import requests
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

toTensor = ToTensor()

class Sentinel2Random(Dataset):
    """
    PyTorch Dataset for the Sentinel-2 Random Split dataset.

    Args:
        root (string, optional): Root directory of dataset where ``sentinel2`` exists.
            Defaults to ``data``.
        train (bool, optional): If True, creates dataset from ``sentinel2/train``,
            otherwise from ``sentinel2/test``.
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
        self.tgt = os.path.join(self.root, "sentinel2", self.label)
        self.tgt_url = self.train_url if train else self.test_url

        # transforms
        self.image_mode = image_mode
        self.transform = transform

        # download if required
        self.download = download
        if download:
            self._download()

        if not self._files_exist(False):
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        # data
        self.images = os.listdir(self.tgt)
        self.size = len(self.images)


    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> dict:
        file_name = self.images[index]
        full_path = os.path.join(self.tgt, file_name)
        img = Image.open(full_path).convert(self.image_mode)
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
        return f"Sentinel2Random(root='{self.root}', train={self.train}, transform={self.transform}, download={self.download})"

    
    # download functions
    def _get_data_size(self):
        in_random_split = False
        response = requests.get(self.meta_url)
        for line in response.text.split('\n'):
            if "Random split (#" in line:
                in_random_split = True
            elif in_random_split and "Sentinel-2" in line:
                line = line.replace("Sentinel-2:", "").replace('\t', '').replace(',', '')
                nums = line.split("=")[0]
                (train, test) = nums.split("/")
                train_size = int(train)
                test_size = int(test)
                if self.train:
                    return train_size
                else:
                    return test_size


    def _files_exist(self, print_ok=True):
        if os.path.exists(self.tgt):
            tgt_size = self._get_data_size()
            current_size = len(os.listdir(self.tgt))
            if current_size != tgt_size:
                if print_ok:
                    print(f"Data exists on device and is size {current_size} but should be {tgt_size}.")
                return False
            else:
                if print_ok:
                    print("Files exist on device and are the correct size.")
                return True
        else:
            if print_ok:
                print("Target path does not exist.")
            return False

    def _download(self):
        # check whether we should download the files and return if already present
        already_exist = self._files_exist()
        if already_exist:
            return
        # make parent folder
        sentinel2_dir = os.path.join(self.root, "sentinel2")
        os.makedirs(sentinel2_dir, exist_ok=True)

        # download zip file to root
        print("Downloading files.")
        zip_path = os.path.join(self.root, f"sentinel2-{self.label}.zip")
        response = requests.get(self.tgt_url, stream=True)
        with open(zip_path, 'wb') as output:
            for chunk in response.iter_content(chunk_size=128):
                output.write(chunk)

        # extract zip to folder
        with ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(self.root)

        # move extracted folder to target
        extracted_folder = os.path.join(self.root, 'sent2')
        if os.path.exists(self.tgt):
            shutil.rmtree(self.tgt)
        shutil.move(extracted_folder, self.tgt)


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
    dataset = Sentinel2Random(**kwargs)
    print(dataset)
    if kwargs["verify"]:
        data = dataset[0]
        print(data)
        data["image"].show()
    print("All good to go, good luck!")
