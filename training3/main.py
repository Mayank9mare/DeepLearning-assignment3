import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
from PIL import Image
import torch.optim as optim
#import config
#from dataset import MapDataset
import numpy as np
from model import Generator
from model import Discriminator
import os
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from torchvision.utils import save_image
import random

config = dict(
    BATCH_SIZE=16,
    IMAGE_HEIGHT= 512,
    IMAGE_WIDTH = 770,
    lr=2e-4,
    EPOCHS=250,
    pin_memory=True,
    num_workers=2,
    gpu_id="6",
    SEED=42,
    return_logs=False,
    L1_LAMDA=100,
    TRAIN="maps/train",
    VAL="maps/val",

)
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"]=config['gpu_id']
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
print(device)
def save_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()


random.seed(config['SEED'])
np.random.seed(config['SEED'])
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.2),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.l=len(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_only_input(image=input_image)["image"]
        target_image = transform_only_mask(image=target_image)["image"]

        return input_image, target_image

import torchvision
import matplotlib.pyplot as plt
def show(img0,k):
  img1=torchvision.utils.make_grid(img0)
  img1=img1/2 +0.5
  img2=img1.numpy()
  plt.imshow(np.transpose(img2, (1, 2, 0)))
  plt.savefig(f'images/img{k}')
tval = {'genloss':[],'discloss':[]}
def train(disc,gen,loader,opt_gen,opt_disc,l1,bce,g_scaler,d_scaler,tval):
    loop =tqdm(loader,leave=True)
    for idx, (x,y) in enumerate(loop):
        x=x.to(device)
        y=y.to(device)

        with torch.cuda.amp.autocast():
            y_fake=gen(x)
            real=disc(x,y)
            loss_real=bce(real,torch.ones_like(real))
            fake=disc(x,y_fake.detach())
            loss_fake=bce(fake,torch.zeros_like(fake))
            disc_loss=(loss_real+loss_fake)/2
            tval['discloss'].append(disc_loss.item())
        disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            fake=disc(x,y_fake)
            gen_fake_loss=bce(fake,torch.ones_like(fake))
            L1_criteria=l1(y_fake,y)*config['L1_LAMDA']
            gen_loss=gen_fake_loss+L1_criteria
            tval['genloss'].append(gen_loss.item())
        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if(idx%10==0):
            loop.set_postfix(
                real=torch.sigmoid(real).mean().item(),
                fake=torch.sigmoid(fake).mean().item()
                )

def main(tval):
    disc=Discriminator().to(device)
    gen=Generator().to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=config['lr'], betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config['lr'], betas=(0.5, 0.999))
    BCE=nn.BCEWithLogitsLoss()
    l1=nn.L1Loss()

    train_dataset=MapDataset(root_dir=config['TRAIN'])
    train_loader=DataLoader(train_dataset,batch_size=config['BATCH_SIZE'],shuffle=True,num_workers=config['num_workers'])
    dataiter = iter(train_loader)

    images, labels = dataiter.next()
    show(images,1)
    
    g_scaler=torch.cuda.amp.GradScaler()
    d_scaler=torch.cuda.amp.GradScaler()
    val_dataset=MapDataset(root_dir=config['VAL'])
    val_loader=DataLoader(val_dataset,batch_size=10,shuffle=True)
    
    for epoch in range(config['EPOCHS']):
        train(disc,gen,train_loader,opt_gen,opt_disc,l1,BCE,g_scaler,d_scaler,tval)
        save_examples(gen,val_loader,epoch,folder="eval4")

main(tval)
        
def loss_curve(tval):
    plt.figure(figsize=(5,4))
    plt.plot(list(range(1,len(tval['genloss'])+1)),tval['genloss'],label='generator-loss')
    plt.plot(list(range(1,len(tval['discloss'])+1)),tval['discloss'],label='discriminator-loss')

    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()
    plt.savefig('loss2')
print(tval)
loss_curve(tval)
    
