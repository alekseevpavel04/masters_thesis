import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models.Net import RRDBNet
from models.Discriminator import VGGStyleDiscriminator
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from collections import OrderedDict

# Определяем базовый путь для model_training относительно корневой директории
BASE_PATH = 'model_training'

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Trainer:
    def __init__(self, train_dataloader, val_dataloader, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        
        # Создаем пути для логов и чекпоинтов внутри model_training
        self.log_dir = os.path.join(BASE_PATH, 'logs', config['log_dir'])
        self.checkpoint_dir = os.path.join(BASE_PATH, 'checkpoints')
        
        # Создаем директории если их нет
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Инициализация моделей
        self.netG = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=config['scale'],
            num_feat=64,
            num_block=23,
            num_grow_ch=32
        ).to(self.device)
        
        self.netD = VGGStyleDiscriminator(
            num_in_ch=3,
            num_feat=64
        ).to(self.device)
        
        # Инициализация весов
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # Оптимизаторы
        self.optimizerG = optim.Adam(
            self.netG.parameters(),
            lr=config['lr_g'],
            betas=(0.9, 0.999)
        )
        self.optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=config['lr_d'],
            betas=(0.9, 0.999)
        )
        
        # Планировщики
        self.schedulerG = optim.lr_scheduler.MultiStepLR(
            self.optimizerG,
            milestones=config['milestones'],
            gamma=0.5
        )
        self.schedulerD = optim.lr_scheduler.MultiStepLR(
            self.optimizerD,
            milestones=config['milestones'],
            gamma=0.5
        )
        
        # Критерии потерь
        self.criterionGAN = nn.BCEWithLogitsLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()
        
        # Веса для различных компонент функции потерь
        self.lambda_adv = config['lambda_adv']
        self.lambda_pixel = config['lambda_pixel']
        self.lambda_feature = config['lambda_feature']
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Счетчики
        self.current_step = 0
        self.current_epoch = 0
        
    def train_step(self, lr_imgs, hr_imgs):
        batch_size = lr_imgs.size(0)
        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)
        
        ############################
        # (1) Обновляем D network
        ############################
        self.netD.zero_grad()
        
        # Обучение на реальных изображениях
        real_output = self.netD(hr_imgs)
        d_loss_real = self.criterionGAN(real_output, real_label)
        
        # Обучение на фейковых изображениях
        fake_imgs = self.netG(lr_imgs)
        fake_output = self.netD(fake_imgs.detach())
        d_loss_fake = self.criterionGAN(fake_output, fake_label)
        
        # Суммарная потеря D
        d_loss = (d_loss_real + d_loss_fake) * 0.5
        d_loss.backward()
        self.optimizerD.step()
        
        ############################
        # (2) Обновляем G network
        ############################
        self.netG.zero_grad()
        
        # Потеря GAN
        fake_output = self.netD(fake_imgs)
        g_loss_gan = self.criterionGAN(fake_output, real_label)
        
        # Потеря пикселей
        g_loss_pixel = self.criterionL1(fake_imgs, hr_imgs)
        
        # Комбинированная потеря
        g_loss = self.lambda_adv * g_loss_gan + self.lambda_pixel * g_loss_pixel
        
        g_loss.backward()
        self.optimizerG.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_loss_gan': g_loss_gan.item(),
            'g_loss_pixel': g_loss_pixel.item()
        }
    
    def validate(self):
        self.netG.eval()
        self.netD.eval()
        val_loss = 0
        psnr_val = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in self.val_dataloader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                fake_imgs = self.netG(lr_imgs)
                val_loss += self.criterionMSE(fake_imgs, hr_imgs).item()
                psnr_val += 10 * np.log10(1 / self.criterionMSE(fake_imgs, hr_imgs).item())
        
        val_loss /= len(self.val_dataloader)
        psnr_val /= len(self.val_dataloader)
        return val_loss, psnr_val
    
    def train(self):
        best_psnr = 0
        for epoch in range(self.config['num_epochs']):
            self.netG.train()
            self.netD.train()
            
            for i, (lr_imgs, hr_imgs) in enumerate(self.train_dataloader):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                losses = self.train_step(lr_imgs, hr_imgs)
                
                # Логирование
                if i % self.config['log_interval'] == 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], "
                          f"Step [{i}/{len(self.train_dataloader)}], "
                          f"D Loss: {losses['d_loss']:.4f}, "
                          f"G Loss: {losses['g_loss']:.4f}")
                    
                    self.writer.add_scalar('Loss/D', losses['d_loss'], self.current_step)
                    self.writer.add_scalar('Loss/G', losses['g_loss'], self.current_step)
                    self.writer.add_scalar('Loss/G_GAN', losses['g_loss_gan'], self.current_step)
                    self.writer.add_scalar('Loss/G_Pixel', losses['g_loss_pixel'], self.current_step)
                
                self.current_step += 1
            
            # Валидация
            val_loss, psnr = self.validate()
            print(f"Validation Loss: {val_loss:.4f}, PSNR: {psnr:.2f}")
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
            self.writer.add_scalar('Validation/PSNR', psnr, epoch)
            
            # Сохранение лучшей модели
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.netG.state_dict(),
                    'discriminator_state_dict': self.netD.state_dict(),
                    'optimizer_g_state_dict': self.optimizerG.state_dict(),
                    'optimizer_d_state_dict': self.optimizerD.state_dict(),
                }, os.path.join(self.checkpoint_dir, 'best_model.pth'))
            
            # Обновление learning rate
            self.schedulerG.step()
            self.schedulerD.step()
            
            self.current_epoch += 1

if __name__ == '__main__':
    # Пример конфигурации
    config = {
        'scale': 2,  # коэффициент масштабирования
        'lr_g': 1e-4,  # learning rate для генератора
        'lr_d': 1e-4,  # learning rate для дискриминатора
        'num_epochs': 100,
        'batch_size': 16,
        'lambda_adv': 0.1,  # вес для adversarial loss
        'lambda_pixel': 1.0,  # вес для pixel loss
        'lambda_feature': 1.0,  # вес для feature loss
        'milestones': [50, 75, 90],  # epochs для уменьшения learning rate
        'log_interval': 100,  # интервал для логирования
        'log_dir': 'realESRGAN'
    }