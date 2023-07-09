import torch
import torch.nn as nn
import numpy
import PIL
import ml_utils
from torch.utils.data import DataLoader
from diffusion_model_model import *
from diffusion_model_settings import *
from diffusion_model_noise_prediction_model import UnetMNISTWithTimeAndContext
from ml_utils import *
import os


def process_folder():
    fols = next(os.walk('./train'))[1]

    for fol in fols:
        files = next(os.walk('./train' + os.sep + fol))[2]

        for file in files:
            os.rename('./train' + os.sep + fol + os.sep + file, fol + '_' + file)


if __name__ == '__main__':
    print('Diffusion model started ...')

    DEVICE = get_device()

    # images_dataset = ml_utils.ImageDatasetUnsupervisedLearning('./train', image_size=(100, 100))

    # train_dataloader = DataLoader(images_dataset, batch_size=64, shuffle=True)

    dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    unet_model = UnetMNISTWithTimeAndContext().to(DEVICE)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=2.6e-06)
    criterion = nn.MSELoss()

    load_torch_model(unet_model, file_name='MNIST_ContextUnetModel_best', path='./saved_models', load_latest=False)
    diffusion_model = DiffusionModel(unet_model, device=DEVICE)

    # diffusion_model.train(train_loader, optimizer, criterion)

    context = torch.zeros(size=(1, 10), dtype=torch.float32).to(DEVICE)
    context[0, 8] = 0.4
    context[0, 9] = 0.3
    context[0, 6] = 0.3
    print(context)
    diffusion_model.backward_process_get_sample(context)
