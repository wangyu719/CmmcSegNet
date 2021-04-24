import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from utils import is_image_file, load_img, save_img
from torchvision import utils as vutils
from Dataset.data import get_test_set

# Testing settings
parser = argparse.ArgumentParser(description='multi-cascade')
parser.add_argument('--dataset', required=True, help='images')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)
root_path = "dataset/"
test_set = get_test_set(root_path + opt.dataset, opt.direction)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)
device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path1 = "checkpoint/{}/netG1_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
model_path2 = "checkpoint/{}/netG2_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
model_path3 = "checkpoint/{}/netG3_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
model_path4 = "checkpoint/{}/netG4_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g1 = torch.load(model_path).to(device)
net_g2 = torch.load(model_path2).to(device)
net_g3 = torch.load(model_path3).to(device)
net_g4 = torch.load(model_path4).to(device)

if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, ), (0.5, ))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    r, g, b = img.split()  
    limg = transform(r)
    input = limg.unsqueeze(0).to(device)
    out = net_g1(input)
    out = net_g2(out)
    out = net_g3(out)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    vutils.save_image(out_img, "result/{}/{}".format(opt.dataset, image_name))  
