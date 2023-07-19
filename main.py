import torch
import torch.nn as nn
import numpy
import PIL
from torch.utils.data import DataLoader
from diffusion_model_model import *
from diffusion_model_settings import *
from diffusion_model_noise_prediction_model import *
from ml_utils import *
import os
import torchvision.datasets as dset


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

    '''
    MNIST
    
    
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
    '''

    # Create the dataset
    dataset = dset.ImageFolder(root='G:\\Guru_Sarath\\Study\\1_Project_PhD\\git_repos\\0_Datasets\\Celeb_Dataset',
                               transform=transforms.Compose([
                                   transforms.Resize(28),
                                   transforms.CenterCrop(28),
                                   transforms.ToTensor(),
                                   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)


    unet_model = Unet28x28WithTime().to(DEVICE)
    unet_model.name = 'UnetCelebAWithTime_28x28'
    #load_torch_model(unet_model, file_name=unet_model.name, path='./saved_models', load_latest=True)

    optimizer = torch.optim.Adam(unet_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    one_sample = (next(iter(train_loader)))[0]

    display_image(one_sample, batch_dim_exist=True)

    diffusion_model = DiffusionModel(unet_model, device=DEVICE)
    diffusion_model.train(train_loader, optimizer, criterion, one_sample)
    diffusion_model.backward_process_get_sample(image_size=(1, 3, 28, 28))




