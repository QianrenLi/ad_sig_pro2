import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import random
import torch.utils.data as data
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
# After training, test and evaluate your model here
from sklearn.metrics import accuracy_score



# In this section, we will apply an CNN to extract features and implement a classification task.
# Firstly, we should build the model by PyTorch. We provide a baseline model here.
# You can use your own model for better performance
class Doubleconv_33(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_33, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_35(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_35, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_37(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_37, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=7),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Tripleconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Tripleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class MLP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, input):
        return self.fc(input)


class Mscnn(nn.Module):
    # TODO: Build a better model
    def __init__(self, ch_in, ch_out):
        super(Mscnn, self).__init__()
        self.conv11 = Doubleconv_33(ch_in, 64)
        self.pool11 = nn.MaxPool1d(3, stride=3)
        self.conv12 = Doubleconv_33(64, 128)
        self.pool12 = nn.MaxPool1d(3, stride=3)
        self.conv13 = Tripleconv(128, 256)
        self.pool13 = nn.MaxPool1d(2, stride=2)
        self.conv14 = Tripleconv(256, 512)
        self.pool14 = nn.MaxPool1d(2, stride=2)
        self.conv15 = Tripleconv(512, 512)
        self.pool15 = nn.MaxPool1d(2, stride=2)

        self.out = MLP(512*27, ch_out)  

    def forward(self, x):
        c11 = self.conv11(x)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        p15 = self.pool15(c15)
        merge = p15.view(p15.size()[0], -1) 
        output = self.out(merge)
        output = F.sigmoid(output)
        return output

# Next, we need to construct the data loader for training. 

# These functions may be helpful for you :)
def data_crop(data_raw, obj_len):
    data_len = np.size(data_raw)
    a = random.randint(0, data_len - obj_len)
    b = a + obj_len
    data_cropped = np.array(data_raw[:, a:b])
    return data_cropped


def data_pad(data_raw, obj_len):
    data_len = np.size(data_raw)
    pad_len = obj_len - data_len
    b = np.zeros((1, pad_len))
    data_padded = np.hstack((data_raw, b))
    return data_padded

class EcgDataset(data.Dataset):
    def __init__(self,  root, data_len, transform=None, target_transform=None):    # transform=x_transform, target_transform=y_transform
        """
        root: the directory of the data 
        data_len: Unknown parameters, but I think it is helpful for you :)
        transform: pre-process for data
        target_transform: target_transform for label
        """
        self.ecgs = []
        self.ecgs = sorted(list(glob(os.path.join(root, '*.mat'))))
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = data_len

    def __getitem__(self, index):
        val_dict_path = self.ecgs[index]  
        val_dict = scio.loadmat(val_dict_path)
        ecg_x = val_dict['value']
        ecg_x_len = np.size(ecg_x)

        # TODO: Note that there may need some pre-process for data with different sizes
        # Write your code here
        if ecg_x_len > self.data_len:
            ecg_x = data_crop(ecg_x, self.data_len)
        else:
            ecg_x = data_pad(ecg_x, self.data_len)
            



        ecg_y = val_dict['label']
        if self.transform is not None:
            ecg_x = self.transform(ecg_x)
            ecg_x = ecg_x.squeeze(dim=1).type(torch.FloatTensor)
        if self.target_transform is not None:
            ecg_y = self.target_transform(ecg_y)

            ecg_y = ecg_y.squeeze(-1).type(torch.FloatTensor)
        return ecg_x, ecg_y

    def __len__(self):
        return len(self.ecgs)
 


# We will use GPU if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Mscnn(1, 1).to(device)   # ch_in, ch_out

# Build pre-processing transformation 
# Note this pre-processing is in PyTorch
x_transforms = transforms.Compose([
        transforms.ToTensor(),  
])
y_transforms = transforms.ToTensor()


# PATH = './weights10_10.pth'
PATH = './pro2_lr_5e-4_epoches_100.pth'

model.load_state_dict(torch.load(PATH))

# Build test dataset
ecg_dataset = EcgDataset('./data/test', 2400, 
                             transform=x_transforms, target_transform=y_transforms)
dataloaders = DataLoader(ecg_dataset, batch_size=1)

# Set model's mode lto eval 
model.eval()

# TODO: add more metrics for evaluation?
# Evaluate 
predict = []
target = []
with torch.no_grad():
    for x, mask in dataloaders:
        y = model(x.to(device))
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        predict.append(torch.squeeze(y).cpu().numpy())
        target.append(torch.squeeze(mask).cpu().numpy())
acc = accuracy_score(target, predict)
print('Accuracy: {}'.format(acc))