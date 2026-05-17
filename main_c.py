import utils
import consts
import models
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

torch.manual_seed(consts.SEED)
random.seed(consts.SEED)
np.random.seed(consts.SEED)

# GPU available?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load model and dataset
model = utils.load_pretrained_cnn(1).to(device)
model.eval()
dataset = utils.TMLDataset(transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(dataset, batch_size=consts.BATCH_SIZE)

# model accuracy
acc_orig = utils.compute_accuracy(model, data_loader, device)
print(f'Model accuracy before flipping: {acc_orig:0.4f}')

# layers whose weights will be flipped
layers = {'conv1': model.conv1,
          'conv2': model.conv2,
          'fc1': model.fc1,
          'fc2': model.fc2,
          'fc3': model.fc3}

# flip bits at random and measure impact on accuracy (via RAD)
RADs_bf_idx = dict([(bf_idx, []) for bf_idx in range(32)])  # will contain a list of RADs for each index of bit flipped
RADs_all = []  # will eventually contain all consts.BF_PER_LAYER*len(layers) RADs
for layer_name in layers:
    layer = layers[layer_name]
    with torch.no_grad():
        W = layer.weight
        W.requires_grad = False
        W_flat = W.view(-1)
        n_weights = W_flat.numel()
        for _ in range(consts.BF_PER_LAYER):
            # Sample a random weight in this layer, flip a random bit in its
            # float32 representation, measure RAD, then restore the weight.
            w_idx = int(np.random.randint(0, n_weights))
            w_orig = float(W_flat[w_idx].item())
            w_flipped, bf_idx = utils.random_bit_flip(w_orig)
            W_flat[w_idx] = float(w_flipped)
            acc_bf = utils.compute_accuracy(model, data_loader, device)
            rad = (acc_orig - acc_bf) / acc_orig
            W_flat[w_idx] = w_orig
            RADs_bf_idx[bf_idx].append(rad)
            RADs_all.append(rad)

# Max and % RAD>15%
RADs_all = np.array(RADs_all)
print(f'Total # weights flipped: {len(RADs_all)}')
print(f'Max RAD: {np.max(RADs_all):0.4f}')
print(f'RAD>15%: {np.sum(RADs_all > 0.15) / RADs_all.size:0.4f}')

# boxplots: bit-flip index vs. RAD
# One box per bit index 0..31 (big-endian IEEE-754: 0=sign, 1-8=exponent,
# 9-31=mantissa) summarising the RAD distribution observed at that position.
plt.figure()
data_per_idx = [RADs_bf_idx[i] if len(RADs_bf_idx[i]) > 0 else [0.0]
                for i in range(32)]
plt.boxplot(data_per_idx, positions=list(range(32)))
plt.xticks(range(32), [str(i) for i in range(32)], fontsize=6)
plt.xlabel('flipped bit index (big-endian)')
plt.ylabel('RAD')
plt.title('RAD distribution per flipped bit index')
plt.tight_layout()
plt.savefig('bf_idx-vs-RAD.jpg')
