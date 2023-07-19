import torch
import torch.nn as nn

DEBUG_MODE = False

'''
/****************************************
           Unet Model Blocks
*****************************************/
'''


class UnetEmbedding(nn.Module):

    def __init__(self, in_size, emb_size):
        super(UnetEmbedding, self).__init__()

        self.embd = nn.Sequential(
            nn.Linear(in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )

    def forward(self, x):
        return self.embd(x)


class Conv2DResBlock(nn.Module):

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int):
        super(Conv2DResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(hid_channels)
        self.activation1 = nn.GELU()

        self.conv2 = nn.Conv2d(hid_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.GELU()

        if self.in_channels != self.out_channels:
            self.conv1x1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.batch_norm1(self.conv1(x))
        x2 = self.activation1(x1)
        x3 = self.conv2(x2)

        if self.in_channels != self.out_channels:
            x_skip = self.conv1x1(x)
        else:
            x_skip = x

        return self.activation2(self.batch_norm2(x_skip + x3))


class UnetDownSamplingBlock(nn.Module):

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int):
        super(UnetDownSamplingBlock, self).__init__()

        self.block = nn.Sequential(
            Conv2DResBlock(in_channels, in_channels, hid_channels),
            Conv2DResBlock(hid_channels, out_channels, out_channels),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class UnetUpSamplingBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(UnetUpSamplingBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            Conv2DResBlock(out_channels, out_channels, out_channels),
            Conv2DResBlock(out_channels, out_channels, out_channels),
        )

    def forward(self, x, x_skip):
        in_tensor = torch.cat((x, x_skip), 1)

        return self.block(in_tensor)


'''
/****************************************
           CFAR10 Unet Models
*****************************************/
'''


class UnetCFAR10(nn.Module):

    def __init__(self, in_size: int = 32):
        super(UnetCFAR10, self).__init__()

        self.name = 'Unet_CFAR10'

        # UNET modules

        # DOWN Sampling path
        self.initial_conv = Conv2DResBlock(in_channels=3, hid_channels=3, out_channels=3)  # (B,3,32,32)
        self.downsample_block_1 = UnetDownSamplingBlock(in_channels=3, hid_channels=128,
                                                        out_channels=128)  # (B,128,16,16)
        self.downsample_block_2 = UnetDownSamplingBlock(in_channels=128, hid_channels=256,
                                                        out_channels=256)  # (B,256,8,8)

        # Bottom part
        flatten_pool_size = (8, 8)  # Same as prev image size
        self.flatten_to_1x1 = nn.AvgPool2d(kernel_size=flatten_pool_size)  # (B,256,1,1)
        self.bring_back_from_flat = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=8,
                                                       stride=8)  # (B,256,8,8)

        # Upsampling path
        self.upsample_block_1 = UnetUpSamplingBlock(in_channels=(256 + 256), out_channels=128)  # (B,128,8,8)
        self.upsample_block_2 = UnetUpSamplingBlock(in_channels=(128 + 128), out_channels=64)  # (B,64,16,16)
        self.final_conv_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)  # (B,32,32,32)
        self.final_conv_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)  # (B,16,32,32)
        self.final_conv_3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)  # (B,3,32,32)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x0 = self.initial_conv(x)
        x1 = self.downsample_block_1(x0)
        x2 = self.downsample_block_2(x1)
        x3 = self.flatten_to_1x1(x2)
        x4 = self.bring_back_from_flat(x3)
        x5 = self.upsample_block_1(x4, x2)
        x6 = self.upsample_block_2(x5, x1)
        x7 = self.act1(self.final_conv_1(x6))
        x8 = self.act1(self.final_conv_2(x7))
        x9 = self.act2(self.final_conv_3(x8))

        return x9


'''
/****************************************
           CelebA Unet Models
*****************************************/
'''


class Unet64x64(nn.Module):

    def __init__(self, in_size: int = 64):
        super(Unet64x64, self).__init__()

        self.name = 'Unet64x64'

        # UNET modules

        # DOWN Sampling path
        self.initial_conv = Conv2DResBlock(in_channels=3, hid_channels=3, out_channels=3)  # (B,3,64,64)
        self.downsample_block_1 = UnetDownSamplingBlock(in_channels=3, hid_channels=64, out_channels=64)  # (B,64,32,32)
        self.downsample_block_2 = UnetDownSamplingBlock(in_channels=64, hid_channels=128,
                                                        out_channels=128)  # (B,128,16,16)
        self.downsample_block_3 = UnetDownSamplingBlock(in_channels=128, hid_channels=256,
                                                        out_channels=256)  # (B,256,8,8)

        # Bottom part
        flatten_pool_size = (8, 8)  # Same as prev image size
        self.flatten_to_1x1 = nn.AvgPool2d(kernel_size=flatten_pool_size)  # (B,256,1,1)
        self.bring_back_from_flat = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=8,
                                                       stride=8)  # (B,256,8,8)

        # Upsampling path
        self.upsample_block_1 = UnetUpSamplingBlock(in_channels=(256 + 256), out_channels=128)  # (B,128,16,16)
        self.upsample_block_2 = UnetUpSamplingBlock(in_channels=(128 + 128), out_channels=64)  # (B,64,32,32)
        self.upsample_block_3 = UnetUpSamplingBlock(in_channels=(64 + 64), out_channels=32)  # (B,32,64,64)
        self.final_conv_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)  # (B,16,64,64)
        self.final_conv_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)  # (B,8,64,64)
        self.final_conv_3 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)  # (B,3,64,64)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x0 = self.initial_conv(x)
        x1 = self.downsample_block_1(x0)
        x2_0 = self.downsample_block_2(x1)
        x2 = self.downsample_block_3(x2_0)
        x3 = self.flatten_to_1x1(x2)
        x4 = self.bring_back_from_flat(x3)
        x5 = self.upsample_block_1(x4, x2)
        x6 = self.upsample_block_2(x5, x2_0)
        x7 = self.upsample_block_3(x6, x1)
        x8 = self.act1(self.final_conv_1(x7))
        x9 = self.act1(self.final_conv_2(x8))
        x10 = self.act2(self.final_conv_3(x9))

        return x10


class Unet64x64WithTime(nn.Module):

    def __init__(self, in_size: int = 64):
        super(Unet64x64WithTime, self).__init__()

        # UNET modules

        self.name = 'Unet64x64WithTime'

        # Time Emb
        self.time_embd_1 = UnetEmbedding(1, 256)
        self.time_embd_2 = UnetEmbedding(1, 128)
        self.time_embd_3 = UnetEmbedding(1, 64)

        # DOWN Sampling path
        self.initial_conv = Conv2DResBlock(in_channels=3, hid_channels=3, out_channels=3)  # (B,3,64,64)
        self.downsample_block_1 = UnetDownSamplingBlock(in_channels=3, hid_channels=64, out_channels=64)  # (B,64,32,32)
        self.downsample_block_2 = UnetDownSamplingBlock(in_channels=64, hid_channels=128,
                                                        out_channels=128)  # (B,128,16,16)
        self.downsample_block_3 = UnetDownSamplingBlock(in_channels=128, hid_channels=256,
                                                        out_channels=256)  # (B,256,8,8)

        # Bottom part
        flatten_pool_size = (8, 8)  # Same as prev image size
        self.flatten_to_1x1 = nn.AvgPool2d(kernel_size=flatten_pool_size)  # (B,256,1,1)
        self.bring_back_from_flat = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=8,
                                                       stride=8)  # (B,256,8,8)

        # Upsampling path
        self.upsample_block_1 = UnetUpSamplingBlock(in_channels=(256 + 256), out_channels=128)  # (B,128,16,16)
        self.upsample_block_2 = UnetUpSamplingBlock(in_channels=(128 + 128), out_channels=64)  # (B,64,32,32)
        self.upsample_block_3 = UnetUpSamplingBlock(in_channels=(64 + 64), out_channels=32)  # (B,32,64,64)
        self.final_conv_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)  # (B,16,64,64)
        self.final_conv_2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)  # (B,8,64,64)
        self.final_conv_3 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)  # (B,3,64,64)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x, t):
        embd_time_1 = self.time_embd_1(t)
        embd_time_2 = self.time_embd_2(t)
        embd_time_3 = self.time_embd_3(t)

        x0 = self.initial_conv(x)
        x1 = self.downsample_block_1(x0)
        x2_0 = self.downsample_block_2(x1)
        x2 = self.downsample_block_3(x2_0)
        x3 = self.flatten_to_1x1(x2)
        x4 = self.bring_back_from_flat(x3)
        embd_time_1 = embd_time_1.view(-1, x4.shape[-3], 1, 1)
        x5 = self.upsample_block_1(x4 + embd_time_1, x2)
        embd_time_2 = embd_time_2.view(-1, x5.shape[-3], 1, 1)
        x6 = self.upsample_block_2(x5 + embd_time_2, x2_0)
        embd_time_3 = embd_time_3.view(-1, x6.shape[-3], 1, 1)
        x7 = self.upsample_block_3(x6 + embd_time_3, x1)
        x8 = self.act1(self.final_conv_1(x7))
        x9 = self.act1(self.final_conv_2(x8))
        x10 = self.act2(self.final_conv_3(x9))

        return x10


class Unet28x28WithTime(nn.Module):

    def __init__(self, context_size=10):
        super(Unet28x28WithTime, self).__init__()

        self.name = 'Unet28x28WithTime'

        # Embedding Layers
        self.time_embd_1 = UnetEmbedding(1, 256)
        self.time_embd_2 = UnetEmbedding(1, 128)

        # UNET modules

        # DOWN Sampling path
        self.initial_conv = Conv2DResBlock(in_channels=3, hid_channels=64, out_channels=64)  # (B,64,28,28)
        self.downsample_block_1 = UnetDownSamplingBlock(in_channels=64, hid_channels=128,
                                                        out_channels=128)  # (B,128,14,14)
        self.downsample_block_2 = UnetDownSamplingBlock(in_channels=128, hid_channels=256,
                                                        out_channels=256)  # (B,256,7,7)

        # Bottom part
        flatten_pool_size = (7, 7)  # Same as prev image size
        self.flatten_to_1x1 = nn.AvgPool2d(kernel_size=flatten_pool_size)  # (B,256,1,1)
        self.bring_back_from_flat = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=7,
                                                       stride=7)  # (B,256,7,7)

        # Upsampling path
        self.upsample_block_1 = UnetUpSamplingBlock(in_channels=(256 + 256), out_channels=128)  # (B,128,8,8)
        self.upsample_block_2 = UnetUpSamplingBlock(in_channels=(128 + 128), out_channels=64)  # (B,64,16,16)
        self.final_conv_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)  # (B,16,32,32)
        self.final_conv_2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1)  # (B,832,32)
        self.final_conv_3 = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, padding=1)  # (B,3,32,32)
        self.act1 = nn.ReLU()

    def forward(self, x, t):
        embd_time_1 = self.time_embd_1(t)
        embd_time_2 = self.time_embd_2(t)

        x0 = self.initial_conv(x)
        if DEBUG_MODE: print('x0 shape', x0.shape)
        x1 = self.downsample_block_1(x0)
        if DEBUG_MODE: print('x1 shape', x1.shape)
        x2 = self.downsample_block_2(x1)
        if DEBUG_MODE: print('x2 shape', x2.shape)
        x3 = self.flatten_to_1x1(x2)
        if DEBUG_MODE: print('x3 shape', x3.shape)
        x4 = self.bring_back_from_flat(x3)
        if DEBUG_MODE: print('x4 shape', x4.shape)

        embd_time_1 = embd_time_1.view(-1, x4.shape[-3], 1, 1)
        x5 = self.upsample_block_1(x4 + embd_time_1, x2)
        if DEBUG_MODE: print('x5 shape', x5.shape)

        embd_time_2 = embd_time_2.view(-1, x5.shape[-3], 1, 1)
        x6 = self.upsample_block_2(x5 + embd_time_2, x1)
        if DEBUG_MODE: print('x6 shape', x6.shape)

        x8 = self.act1(self.final_conv_1(x6))
        if DEBUG_MODE: print('x8 shape', x8.shape)
        x9 = self.act1(self.final_conv_2(x8))
        if DEBUG_MODE: print('x9 shape', x9.shape)
        x10 = self.final_conv_3(x9)
        if DEBUG_MODE: print('x10 shape', x10.shape)

        return x10


'''
/****************************************
           MNIST Unet Models
*****************************************/
'''


class UnetMNISTWithTimeAndContext(nn.Module):

    def __init__(self, context_size=10):
        super(UnetMNISTWithTimeAndContext, self).__init__()

        self.name = 'Unet_MNIST_with_time_and_context'

        # Embedding Layers
        self.context_embd_1 = UnetEmbedding(context_size, 256)
        self.context_embd_2 = UnetEmbedding(context_size, 128)
        self.time_embd_1 = UnetEmbedding(1, 256)
        self.time_embd_2 = UnetEmbedding(1, 128)

        # UNET modules

        # DOWN Sampling path
        self.initial_conv = Conv2DResBlock(in_channels=1, hid_channels=3, out_channels=3)  # (B,1,28,28)
        self.downsample_block_1 = UnetDownSamplingBlock(in_channels=3, hid_channels=128,
                                                        out_channels=128)  # (B,128,14,14)
        self.downsample_block_2 = UnetDownSamplingBlock(in_channels=128, hid_channels=256,
                                                        out_channels=256)  # (B,256,7,7)

        # Bottom part
        flatten_pool_size = (7, 7)  # Same as prev image size
        self.flatten_to_1x1 = nn.AvgPool2d(kernel_size=flatten_pool_size)  # (B,256,1,1)
        self.bring_back_from_flat = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=7,
                                                       stride=7)  # (B,256,7,7)

        # Upsampling path
        self.upsample_block_1 = UnetUpSamplingBlock(in_channels=(256 + 256), out_channels=128)  # (B,128,8,8)
        self.upsample_block_2 = UnetUpSamplingBlock(in_channels=(128 + 128), out_channels=64)  # (B,64,16,16)
        self.final_conv_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)  # (B,16,32,32)
        self.final_conv_2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=3, padding=1)  # (B,832,32)
        self.final_conv_3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)  # (B,3,32,32)
        self.act1 = nn.ReLU()

    def forward(self, x, t, context_vec):
        embd_context_1 = self.context_embd_1(context_vec)
        embd_context_2 = self.context_embd_2(context_vec)
        embd_time_1 = self.time_embd_1(t)
        embd_time_2 = self.time_embd_2(t)

        x0 = self.initial_conv(x)
        if DEBUG_MODE: print('x0 shape', x0.shape)
        x1 = self.downsample_block_1(x0)
        if DEBUG_MODE: print('x1 shape', x1.shape)
        x2 = self.downsample_block_2(x1)
        if DEBUG_MODE: print('x2 shape', x2.shape)
        x3 = self.flatten_to_1x1(x2)
        if DEBUG_MODE: print('x3 shape', x3.shape)
        x4 = self.bring_back_from_flat(x3)
        if DEBUG_MODE: print('x4 shape', x4.shape)

        embd_context_1 = embd_context_1.view(-1, x4.shape[-3], 1, 1)
        embd_time_1 = embd_time_1.view(-1, x4.shape[-3], 1, 1)
        x5 = self.upsample_block_1(x4 * embd_context_1 + embd_time_1, x2)
        if DEBUG_MODE: print('x5 shape', x5.shape)

        embd_context_2 = embd_context_2.view(-1, x5.shape[-3], 1, 1)
        embd_time_2 = embd_time_2.view(-1, x5.shape[-3], 1, 1)
        x6 = self.upsample_block_2(x5 * embd_context_2 + embd_time_2, x1)
        if DEBUG_MODE: print('x6 shape', x6.shape)

        x8 = self.act1(self.final_conv_1(x6))
        if DEBUG_MODE: print('x8 shape', x8.shape)
        x9 = self.act1(self.final_conv_2(x8))
        if DEBUG_MODE: print('x9 shape', x9.shape)
        x10 = self.final_conv_3(x9)
        if DEBUG_MODE: print('x10 shape', x10.shape)

        return x10


'''
/****************************************
           Testing
*****************************************/
'''

if __name__ == '__main__':
    print('Testing Context Unet...')

    x_mnist = torch.randn((1, 1, 28, 28))
    x_celeb_a = torch.randn((1, 3, 64, 64))
    x_cfar10 = torch.randn((1, 3, 32, 32))

    res_block = Conv2DResBlock(3, 3, 3)
    down_block = UnetDownSamplingBlock(3, 4, 8)
    down_block2 = UnetDownSamplingBlock(8, 8, 16)

    with torch.no_grad():
        x0 = res_block(x_cfar10)
        print('x0 shape ', x0.shape)

        x1 = down_block(x_cfar10)
        print('x1 shape ', x1.shape)

        x2 = down_block2(x1)
        print('x2 shape ', x2.shape)

        pool_size = (x2.shape[-2], x2.shape[-1])
        print(pool_size)
        x3 = nn.AvgPool2d(kernel_size=pool_size)(x2)
        print('x3(flat) shape ', x3.shape)

        num_channels = x3.shape[-3]
        kernel_size = x2.shape[-2]
        x4 = nn.ConvTranspose2d(num_channels, num_channels, kernel_size=kernel_size, stride=5)(x3)
        print('x4 shape ', x4.shape)

        x5 = UnetUpSamplingBlock(x4.shape[-3] + x2.shape[-3], 10)(x4, x2)
        print('x5 shape ', x5.shape)

        x6 = UnetUpSamplingBlock(x5.shape[-3] + x1.shape[-3], 10)(x5, x1)
        print('x6 shape ', x6.shape)

        x7 = UnetUpSamplingBlock(x6.shape[-3] + x0.shape[-3], 10)(x6, x0)
        print('x7 shape ', x7.shape)

        print('------------------------------------------')

        x_out = UnetCFAR10()(x_cfar10)
        print(x_out.shape)

        x_out = UnetMNISTWithTimeAndContext(5)(x_mnist, torch.tensor([[4.0]]),
                                               torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]))
        print(x_out.shape)

        x_out = Unet64x64()(x_celeb_a)
        print(x_out.shape)

        x_out = Unet64x64WithTime()(x_celeb_a, torch.tensor([[4.0]]))
        print(x_out.shape)
