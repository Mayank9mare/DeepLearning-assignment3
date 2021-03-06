# -*- coding: utf-8 -*-
"""DCGAN

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d0tNyu4A9pQakQjQmGs3BOL1L5ILXaWA
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import random

import torchvision.models as models

model=models.resnet18()
model2=nn.Sequential(*(list(model.children())[:-1]))
print(model2.children)
config = dict(
    BATCH_SIZE=64,
    IMAGE_HEIGHT= 512,
    IMAGE_WIDTH = 770,
    lr=1e-5,
    EPOCHS=20,
    pin_memory=True,
    num_workers=2,
    gpu_id="4",
    SEED=42,
    return_logs=False,
)
random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
class Discriminator(nn.Module):
    def __init__(self, channels, features=512):
        super(Discriminator, self).__init__()
        self.disc=nn.Sequential(
            nn.Conv2d(1,3,kernel_size=1,stride=1,padding=0),
            nn.LeakyReLU(0.2),
            model2,
            nn.Conv2d( 512, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),

        )
            
        

    def forward(self, x):
        a=self.disc(x)
        #a=self.sg(a)
        #print(a.shape)
        return a


class Generator(nn.Module):
    def __init__(self, noise, channels_img, f):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            #N x channels_noise x 1 x 1
            self.block_gen(noise, f * 16, 4, 1, 0),  # img: 4x4
            self.block_gen(f * 16, f * 8, 4, 2, 1),  # img: 8x8
            self.block_gen(f* 8, f* 4, 4, 2, 1),  # img: 16x16
            self.block_gen(f* 4, f* 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                f * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            #N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def block_gen(self, in_ch, out, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch,out,kernel_size,stride,padding,bias=False,),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


def init_weights(m):
    for x in m.modules():
        if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(x.weight.data, 0.0, 0.02)

c=0

def show(img0,k):
  img1=torchvision.utils.make_grid(img0)
  img1=img1/2 +0.5
  img2=img1.numpy()
  plt.imshow(np.transpose(img2, (1, 2, 0)))
  plt.savefig(f'images/img{k}')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_id']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
LEARNING_RATE = config['lr']
BATCH_SIZE = config['BATCH_SIZE']
IMAGE_SIZE = 64
CHANNELS_IMG = 1
NOISE_DIM = 100
EPOCHS = config['EPOCHS']
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms,
                       download=True)


#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,pin_memory=config['pin_memory'],num_workers=config['num_workers'])
gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
init_weights(gen)
init_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()
k=0
tval = {'genloss':[],'discloss':[]}
for epoch in range(EPOCHS):
    print(epoch,"starting")
    loss1=0
    loss2=0
    l=len(dataloader)
  
    for batch_idx, (real, _) in enumerate(dataloader):
        
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)
        

        #Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        loss2=(loss_disc.item())/l
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        #Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        loss1=(loss_gen.item())/l
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        tval['discloss'].append(float(loss2))
        tval['genloss'].append(float(loss1))

        
        if batch_idx % 100 == 0:
            k+=1
            show(fake.to('cpu'),k)
            k+=1
            show(real.to('cpu'),k)
            print(f"Epoch [{epoch}/{EPOCHS}] Batch {batch_idx}/{len(dataloader)} \Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")

            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1
    

def loss_curve(tval):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,len(tval['genloss'])+1)),tval['genloss'],label='generator-loss')
    plt.plot(list(range(1,len(tval['discloss'])+1)),tval['discloss'],label='discriminator-loss')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('loss')
print(tval)
loss_curve(tval)