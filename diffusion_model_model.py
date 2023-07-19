import torch
import torch.nn as nn
import numpy as np
import PIL
from ml_utils import *
from diffusion_model_settings import *
from tqdm import tqdm
import random
from PIL import Image

class DiffusionModel():

    def __init__(self, noise_prediction_model, T=500, time_schedule='linear', device='cpu'):

        self.device = device
        self.time_start = 0
        self.time_end = T
        self.num_time_steps = self.time_end - self.time_start

        self.model = noise_prediction_model

        if time_schedule == 'linear':
            self.beta_t = (0.02 - 1e-4) * torch.linspace(0, 1, T + 1).to(self.device)
        else:
            print('Unrecognized time schedule !! Switching to linear time schedule ')
            self.beta_t = torch.linspace(0, 1, T + 1).to(self.device)

        self.sqrt_beta_t = torch.sqrt(self.beta_t)
        self.alpha_t = 1 - self.beta_t
        self.alpha_bar_t = torch.cumprod(self.alpha_t, dim=0)
        self.sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t)
        self.sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t)

    def forward_process_add_noise(self, input_images_t0, t):
        standard_normal_noise = torch.randn_like(input_images_t0, device=self.device)
        noisy_image_t = (input_images_t0 * self.sqrt_alpha_bar_t[t]) + (
                standard_normal_noise * self.sqrt_1_minus_alpha_bar_t[t])

        return noisy_image_t, standard_normal_noise

    def backward_process_remove_noise(self, input_images_t, t, pred_noise, noise_to_prevent_collapse=None):
        if noise_to_prevent_collapse is None:
            noise_to_prevent_collapse = torch.randn_like(input_images_t)
        noise = self.sqrt_beta_t[t] * noise_to_prevent_collapse
        mean = (input_images_t - pred_noise * ((1 - self.alpha_t[t]) / (1 - self.alpha_bar_t[t]).sqrt())) / \
               self.alpha_t[t].sqrt()
        return mean + noise, mean

    @torch.no_grad()
    def backward_process_get_sample(self, image_size=(1, 3, 64, 64), interactive=True, save_intermediate_outputs=False):

        self.model.eval()

        image_T = torch.randn(image_size).to(self.device)
        image_t = image_T

        ones_ = torch.ones(size=(1, 1), device=self.device)

        for t in tqdm(range(self.time_end, self.time_start, -1)):
            time_step = ones_ * (t / self.num_time_steps)

            if t == 1:
                noise_to_prevent_collapse = 0
            else:
                noise_to_prevent_collapse = torch.randn_like(image_t)

            pred_noise = self.model(image_t, time_step)
            image_t, no_noise_image = self.backward_process_remove_noise(image_t, t, pred_noise, noise_to_prevent_collapse)

            if save_intermediate_outputs:
                transform_to_pil = get_torch_transforms_torch_image_to_pil_image()
                transform_to_pil = transforms.Compose(transform_to_pil)

                pil_image = transform_to_pil(no_noise_image.detach().squeeze(dim=0).cpu())
                pil_image.save(f'./saved_images/saved_image_{t}.png')


            if interactive and (t == 10 or t == 1 or t%100 == 0):
                print('t = ', t)
                image_disp = image_t.detach().squeeze(dim=0).to('cpu')
                #for i in range(3):
                    #max_p = torch.max(image_disp[i])
                    #min_p = torch.min(image_disp[i])

                    #image_disp[i] = (image_disp[i] - min_p) / (max_p - min_p)

                display_image(image_disp)


    # For batch size of 25
    def backward_process_get_sample_2(self, save_intermediate_outputs=True):

        self.model.eval()

        image_T = torch.randn((25, 1, 28, 28)).to(self.device)
        image_t = image_T

        ones_ = torch.ones(size=(1, 1), device=self.device)

        for t in tqdm(range(self.time_end, self.time_start, -1)):
            time_step = ones_ * (t / self.num_time_steps)

            if t == 1:
                noise_to_prevent_collapse = 0;
            else:
                noise_to_prevent_collapse = torch.randn_like(image_t)

            pred_noise = self.model(image_t, time_step)
            image_t, no_noise_image = self.backward_process_remove_noise(image_t, t, pred_noise, noise_to_prevent_collapse)

            if save_intermediate_outputs:
                pass

            if t == 1:
                save_tensor_images(no_noise_image)


    def train(self, dataloader, optimizer, criterion, train_one_sample):

        model = self.model

        model.train()

        for epoch in range(EPOCHS):

            optimizer.param_groups[0]['lr'] = 0.00001*(1-epoch/EPOCHS)
            print(f'Learning rate = {0.00001*(1-epoch/EPOCHS)}')

            iter_index = 0
            loss_sum = 0
            for images, _ in tqdm(dataloader):

                if train_one_sample is not None:
                    images = train_one_sample.to(self.device)
                else:
                    images = images.to(self.device)

                batch_size = images.shape[0]

                # Get a random timestep
                time_step_int = random.randint(1, self.time_end)
                time_step = torch.ones(size=(batch_size, 1), device=self.device) * (time_step_int / self.num_time_steps)

                # Get the noisy images and the noise added to it
                noisy_images, expected_noise = self.forward_process_add_noise(images, time_step_int)

                # Run the noisy images through the model and update parameters
                optimizer.zero_grad()
                noise_pred = model(noisy_images, time_step)
                loss = criterion(noise_pred, expected_noise)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                if iter_index != 0 and iter_index % 20 == 0:
                    print(f'loss = {loss_sum / 200}')
                    loss_sum = 0

                if iter_index % 20000 == 0:
                    optimizer.param_groups[0]['lr'] = 0.00001 * (1 - epoch / EPOCHS)

                if iter_index % 1000 == 0:
                    save_torch_model(model, file_name=model.name, additional_info=f'_{epoch}_{iter_index}', two_copies=True)

                iter_index += 1

            save_torch_model(model, file_name=model.name, additional_info=f'_{epoch}_end', two_copies=True)


class Conditional_DiffusionModel(DiffusionModel):

    def __init__(self, noise_prediction_model, T=500, time_schedule='linear', device='cpu'):

        super()

    def backward_process_get_sample(self, context=None, interactive=False, save_intermediate_outputs=True):

        self.model.eval()

        image_T = torch.randn((1, 1, 28, 28)).to(self.device)
        image_t = image_T

        ones_ = torch.ones(size=(1, 1), device=self.device)

        for t in tqdm(range(self.time_end, self.time_start, -1)):
            time_step = ones_ * (t / self.num_time_steps)

            if t == 1:
                noise_to_prevent_collapse = 0;
            else:
                noise_to_prevent_collapse = torch.randn_like(image_t)

            pred_noise = self.model(image_t, time_step, context)
            image_t, no_noise_image = self.backward_process_remove_noise(image_t, t, pred_noise, noise_to_prevent_collapse)

            if save_intermediate_outputs:
                transform_to_pil = get_torch_transforms_torch_image_to_pil_image()
                transform_to_pil = transforms.Compose(transform_to_pil)

                min_pix_val = torch.min(no_noise_image).item()

                if min_pix_val < 0:
                    no_noise_image = (no_noise_image + abs(min_pix_val))
                else:
                    no_noise_image = (no_noise_image + abs(min_pix_val))

                max_pix_val = torch.max(no_noise_image).item()
                no_noise_image = no_noise_image / max_pix_val

                if t == 1:
                    threshold = torch.max(no_noise_image).item() / 2
                    no_noise_image[no_noise_image < threshold] = 0
                    no_noise_image[no_noise_image > threshold] = 1

                pil_image = transform_to_pil(no_noise_image.detach().squeeze(dim=0).cpu())
                pil_image.save(f'./saved_images/saved_image_{t}.png')


            if interactive and (t == 10 or t == 1 or t%100 == 0):
                print('t = ', t)
                display_image(image_t.detach().squeeze(dim=0).to('cpu'))


    def backward_process_get_sample_2(self, context=None, save_intermediate_outputs=True):

        self.model.eval()

        image_T = torch.randn((25, 1, 28, 28)).to(self.device)
        image_t = image_T

        ones_ = torch.ones(size=(1, 1), device=self.device)

        for t in tqdm(range(self.time_end, self.time_start, -1)):
            time_step = ones_ * (t / self.num_time_steps)

            if t == 1:
                noise_to_prevent_collapse = 0;
            else:
                noise_to_prevent_collapse = torch.randn_like(image_t)

            pred_noise = self.model(image_t, time_step, context)
            image_t, no_noise_image = self.backward_process_remove_noise(image_t, t, pred_noise, noise_to_prevent_collapse)

            if save_intermediate_outputs:
                pass

            if t == 1:
                save_tensor_images(no_noise_image)


    def train(self, dataloader, optimizer, criterion):

        model = self.model

        model.train()

        for epoch in range(EPOCHS):

            optimizer.param_groups[0]['lr'] = 0.00001*(1-epoch/EPOCHS)
            print(f'Learning rate = {0.00001*(1-epoch/EPOCHS)}')

            iter_index = 0
            loss_sum = 0
            for images, context in tqdm(dataloader):
                images = images.to(self.device)
                context = torch.Tensor(get_numpy_onehot_array(context.numpy(), num_categories=10)).to(self.device)

                batch_size = images.shape[0]

                # Get a random timestep
                time_step_int = random.randint(1, self.time_end)
                time_step = torch.ones(size=(batch_size, 1), device=self.device) * (time_step_int / self.num_time_steps)

                # Get the noisy images and the noise added to it
                noisy_images, expected_noise = self.forward_process_add_noise(images, time_step_int)

                # Run the noisy images through the model and update parameters
                optimizer.zero_grad()
                noise_pred = model(noisy_images, time_step, context)
                loss = criterion(noise_pred, expected_noise)
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

                if iter_index != 0 and iter_index % 20 == 0:
                    print(f'loss = {loss_sum / 200}')
                    loss_sum = 0

                if iter_index % 1000 == 0:
                    save_torch_model(model, file_name='MNIST_ContextUnetModel',
                                     additional_info=f'_{epoch}_{iter_index}', two_copies=True)

                iter_index += 1

            save_torch_model(model, file_name='MNIST_ContextUnetModel', additional_info=f'_{epoch}_end', two_copies=True)




if __name__ == '__main__':
    print('Sanity Testing .....')

    img = get_image_tensor('test_img.jpg')

    noisy_img, _ = Conditional_DiffusionModel(None).forward_process_add_noise(img, 20)
    display_image(noisy_img)
