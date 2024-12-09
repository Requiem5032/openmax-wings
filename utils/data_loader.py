import pandas as pd
import torch
import utils.constants as const

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import transform
from utils.data_loader_utils import *
from utils.preprocess import alb_transform_train, alb_transform_test


class MosDataset(Dataset):
    def __init__(self, img_size, data_df, transformer=None):
        super().__init__()
        self.img_size = img_size
        self.transformer = transformer
        self.images_df = data_df

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        imagename = self.images_df.loc[idx, 'Path']

        image = load_image(imagename)
        y0 = int(0.2*image.shape[0])
        y1 = int(0.8*image.shape[0])
        image = image[y0:y1, :]
        image = make_square(image)

        label = self.images_df['Species_ID'][idx]

        if self.transformer:
            image = self.transformer(image=image)['image']

        else:
            image = transform.resize(
                image, (self.img_size, self.img_size))

        image = torch.from_numpy(image).permute(-1, 0, 1).float()

        return image, label


def get_data_loaders(data_df):
    train_df = data_df[data_df['Split'] == 'Train'].reset_index(drop=True)
    val_df = data_df[data_df['Split'] == 'Val'].reset_index(drop=True)

    train_tf = alb_transform_train(const.IMG_SIZE, p=const.AUGMENT_PROB)
    val_tf = alb_transform_test(const.IMG_SIZE)

    # set up the datasets
    train_ds = MosDataset(
        img_size=const.IMG_SIZE,
        data_df=train_df,
        transformer=train_tf,
    )

    val_ds = MosDataset(
        img_size=const.IMG_SIZE,
        data_df=val_df,
        transformer=val_tf,
    )

    train_sampler = SubsetRandomSampler(range(len(train_ds)))

    # set up the data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_dl, val_dl


def get_thresh_data_loader(data_df):
    train_df = data_df[data_df['Split'] == 'Train'].reset_index(drop=True)
    train_df = train_df.drop('Split', axis=1)
    unk_df = pd.read_csv('data/splits/unknown_data.csv')
    test_df = pd.concat([train_df, unk_df], ignore_index=True)
    test_tf = alb_transform_test(const.IMG_SIZE)

    # set up the datasets
    test_dataset = MosDataset(
        img_size=const.IMG_SIZE,
        data_df=test_df,
        transformer=test_tf,
    )

    # set up the data loader
    test_dl = DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return test_dl


def get_thresh_data_loader(data_df):
    val_df = data_df[data_df['Split'] == 'Val'].reset_index(drop=True)
    val_df = val_df.drop('Split', axis=1)
    unk_df = pd.read_csv('data/splits/unknown_data.csv')
    test_df = pd.concat([val_df, unk_df], ignore_index=True)
    test_tf = alb_transform_test(const.IMG_SIZE)

    # set up the datasets
    test_dataset = MosDataset(
        img_size=const.IMG_SIZE,
        data_df=test_df,
        transformer=test_tf,
    )

    # set up the data loader
    test_dl = DataLoader(
        test_dataset,
        batch_size=const.BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return test_dl
