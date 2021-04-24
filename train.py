import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from utils import is_image_file, load_img, save_img
from torchvision import utils as vutils
from unet_g import define_G, define_D, get_scheduler, update_learning_rate
from Dataset.data import get_training_set
from loss.ganloss import GANLoss
from loss.vggloss import *

parser = argparse.ArgumentParser(description='multi-cascade')
parser.add_argument('--dataset', required=True, help='images')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')
parser.add_argument('--outf', default='./tranre', help='folder to output images and model checkpoints')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
opt = parser.parse_args()


print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)


device = torch.device("cuda:0" if opt.cuda else "cpu")

print('===> Building models')

net_g1 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_g2 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_g3 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)


net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)

vgg_loss = VGGLoss().to(device)
criterionGAN = GANLoss(opt.gan_mode).to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

optimizer_g = optim.Adam(net_g1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g2 = optim.Adam(net_g2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g3 = optim.Adam(net_g3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

net_g1_scheduler = get_scheduler(optimizer_g, opt)
net_g2_scheduler = get_scheduler(optimizer_g2, opt)
net_g3_scheduler = get_scheduler(optimizer_g3, opt)

net_d_scheduler = get_scheduler(optimizer_d, opt)


for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        fake_b1 = net_g1((real_a))
        fake_b2 = net_g2(fake_b1)
        fake_b3 = net_g3(fake_b2)

        ######################
        # (1) Update D network
        ######################
        optimizer_d.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b1), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        loss_d_1 = (loss_d_fake + loss_d_real) * 0.5
        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()
        optimizer_g2.zero_grad()
        optimizer_g3.zero_grad()
        fake_ab = torch.cat((real_a, fake_b1), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
        loss_g_l1 = criterionL1(fake_b1, real_b) * opt.lamb

        loss_v = vgg_loss(fake_b1, real_b)
        loss_g_1 = loss_g_gan + loss_g_l1 + loss_v
        ######################
        # (2) Update D network
        ######################
        
        # # train with fake
        fake_ab2 = torch.cat((real_a, fake_b2), 1)
        pred_fake2 = net_d.forward(fake_ab2.detach())
        loss_d_fake2 = criterionGAN(pred_fake2, False)

        # # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real2 = net_d.forward(real_ab)
        loss_d_real2 = criterionGAN(pred_real2, True)

        loss_d_2 = (loss_d_fake2 + loss_d_real2) * 0.5

        # # ######################
        #  (2) Update G2 network
        # # ######################
        fake_ab2 = torch.cat((real_a, fake_b2), 1)
        pred_fake2 = net_d.forward(fake_ab2)
        loss_g_gan_2 = criterionGAN(pred_fake2, True)
        loss_g_l1_2 = criterionL1(fake_b2, real_b) * opt.lamb
        loss_v_2 = vgg_loss(fake_b2, real_b)
        loss_g_2 = loss_g_gan_2 + loss_g_l1_2 + loss_v_2
        # ######################
        # # (3) Update D network
        # ######################
        fake_ab3 = torch.cat((real_a, fake_b3), 1)
        pred_fake3 = net_d.forward(fake_ab3.detach())
        loss_d_fake3 = criterionGAN(pred_fake3, False)
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real3 = net_d.forward(real_ab)
        loss_d_real3 = criterionGAN(pred_real, True)
        loss_d_3 = (loss_d_fake3 + loss_d_real3) * 0.5
        loss_d = loss_d_1+loss_d_2+loss_d_3
        loss_d.backward(retain_graph=True)
        optimizer_d.step()
        # # # ######################
        #  (2) Update G3 network
        # # # ######################
        fake_ab3 = torch.cat((real_a, fake_b3), 1)
        pred_fake3 = net_d.forward(fake_ab3)
        loss_g_gan_3 = criterionGAN(pred_fake3, True)

        # # # # Second, G(A) = B
        loss_g_l1_3 = criterionL1(fake_b3, real_b) * opt.lamb
        loss_v_3 = vgg_loss(fake_b3, real_b)
        loss_g_3 = loss_g_gan_3 + loss_g_l1_3 + loss_v_3 
        loss_g = 0.1667*loss_g_1 + 0.3333*loss_g_2 + 0.5*loss_g_3 
        loss_g.backward(retain_graph=True)

        optimizer_g.step()
        optimizer_g2.step()
        optimizer_g3.step()

        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} ".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        
    update_learning_rate(net_g1_scheduler, optimizer_g)
    update_learning_rate(net_g2_scheduler, optimizer_g2)
    update_learning_rate(net_g3_scheduler, optimizer_g3)
    update_learning_rate(net_d_scheduler, optimizer_d)
    #checkpoint
    if epoch % 1 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
        net_g1_model_out_path = "checkpoint/{}/netG1_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_g2_model_out_path = "checkpoint/{}/netG2_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_g3_model_out_path = "checkpoint/{}/netG3_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
        torch.save(net_g1, net_g1_model_out_path)
        torch.save(net_g2, net_g2_model_out_path)
        torch.save(net_g3, net_g3_model_out_path)
        torch.save(net_d, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))


