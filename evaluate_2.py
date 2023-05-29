import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
from torch import nn
from model import TrafficSignNet
from data import get_test_loader,get_test_loader_adv
from torchvision.utils import make_grid
from train import valid_batch,valid_batch_Adv
from tqdm import tqdm


def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(
            *[valid_batch(model, loss_func, x, y) for x, y in tqdm(dl)])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%")

def evaluateAdv(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(
            *[valid_batch_Adv(model, loss_func, x, y) for x, y in tqdm(dl)])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%")

def convert_image_np(img):
    img = img.numpy().transpose((1, 2, 0)).squeeze()
    return img


def visualize_stn(dl, outfile):
    with torch.no_grad():
        data = next(iter(dl))[0]

        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        input_grid = convert_image_np(make_grid(input_tensor))
        transformed_grid = convert_image_np(make_grid(transformed_tensor))

        # Plot the results side-by-side
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((16, 16))
        ax[0].imshow(input_grid)
        ax[0].set_title('Dataset Images')
        ax[0].axis('off')

        ax[1].imshow(transformed_grid)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')

        plt.savefig(outfile)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('./model_adv/modeladv.pt', map_location=device)
checkpoint2 = torch.load('./model_benign/model.pt', map_location=device)
# Neural Network and Loss Function
model = TrafficSignNet().to(device)
model.load_state_dict(checkpoint)
# model.cuda()
model.eval()
criterion = nn.CrossEntropyLoss()

model2 = TrafficSignNet().to(device)
model2.load_state_dict(checkpoint2)
# model2.cuda()
model2.eval()
criterion2 = nn.CrossEntropyLoss()

# Data Initialization and Loading
test_loader = get_test_loader('./data', device)
test_loader_adv = get_test_loader_adv('./data', device)
# print('Evaluating Using Adv data and none-mitigating model')

# evaluate(model2, criterion2, test_loader)
# print('Evaluating Using Adv data and mitigating model')

# evaluate(model, criterion, test_loader) 
model.eval()

losses, corrects, nums = zip(*[valid_batch(model2, criterion, x, y) for x, y in tqdm(test_loader_adv)])
test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
test_accuracy = np.sum(corrects) / np.sum(nums) * 100
print(f"Test loss: {test_loss:.6f}\t"
  f"Test accruacy: {test_accuracy:.3f}%")
# visualize_stn(test_loader, args.outfile)

model2.eval()

losses, corrects, nums = zip(*[valid_batch(model, criterion, x, y) for x, y in tqdm(test_loader_adv)])
test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
test_accuracy = np.sum(corrects) / np.sum(nums) * 100
print(f"Test loss: {test_loss:.6f}\t"
  f"Test accruacy: {test_accuracy:.3f}%")
# visualize_stn(test_loader, '/Users/abdullatif/Desktop/codes/DrAla/traffic-sign-recognition/output/visualize_stn.png')