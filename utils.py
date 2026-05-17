import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    # Iterate over the loader, take argmax of logits, return fraction correct.
    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for data in data_loader:
            x, y = data[0].to(device), data[1].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)
    return n_correct / n_total if n_total > 0 else 0.0


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    # If targeted: draw t = (c_x + randint(1, n_classes)) % n_classes (=> t != c_x);
    # run the attack per-batch and concatenate (x_adv, y_used) across the dataset.
    x_adv_all = []
    y_all = []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            offset = torch.randint(1, n_classes, y.shape, device=device)
            y_use = (y + offset) % n_classes
        else:
            y_use = y
        x_adv = attack.execute(x, y_use, targeted=targeted)
        x_adv_all.append(x_adv.detach())
        y_all.append(y_use.detach())
    return torch.cat(x_adv_all, dim=0), torch.cat(y_all, dim=0)


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    # Same flow as run_whitebox_attack, but the black-box attack also returns
    # a per-sample query count which we accumulate alongside the samples.
    x_adv_all = []
    y_all = []
    n_queries_all = []
    for data in data_loader:
        x, y = data[0].to(device), data[1].to(device)
        if targeted:
            offset = torch.randint(1, n_classes, y.shape, device=device)
            y_use = (y + offset) % n_classes
        else:
            y_use = y
        x_adv, n_queries = attack.execute(x, y_use, targeted=targeted)
        x_adv_all.append(x_adv.detach())
        y_all.append(y_use.detach())
        n_queries_all.append(n_queries.detach())
    return (torch.cat(x_adv_all, dim=0),
            torch.cat(y_all, dim=0),
            torch.cat(n_queries_all, dim=0))


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    # Mini-batch the precomputed adversarial samples through the model and
    # count preds == y (targeted) / preds != y (untargeted), divide by total.
    model.eval()
    n_success = 0
    n_total = x_adv.size(0)
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            x_b = x_adv[start:end].to(device)
            y_b = y[start:end].to(device)
            preds = model(x_b).argmax(dim=1)
            if targeted:
                n_success += (preds == y_b).sum().item()
            else:
                n_success += (preds != y_b).sum().item()
    return n_success / n_total if n_total > 0 else 0.0


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    # Pack as big-endian IEEE-754 single precision (struct '!f' = 4 bytes) and
    # turn each byte into its 8-bit binary string => 32 characters total.
    packed = struct.pack('!f', float(num))
    bits = ''.join(f'{byte:08b}' for byte in packed)
    return bits


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    # Inverse of binary(): split the 32-char string into 4 bytes (8 bits each)
    # and unpack as a big-endian float32 ('!f').
    if len(binary) != 32:
        raise ValueError('binary string must be exactly 32 characters long')
    bytes_ = bytes(int(binary[i:i + 8], 2) for i in range(0, 32, 8))
    return struct.unpack('!f', bytes_)[0]


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    # Encode the weight as 32 IEEE-754 bits, flip one uniformly-random bit,
    # decode back to float32; returns (new_w, bf_idx) for the RAD analysis.
    bits = binary(w)
    bf_idx = int(np.random.randint(0, 32))
    flipped_char = '1' if bits[bf_idx] == '0' else '0'
    new_bits = bits[:bf_idx] + flipped_char + bits[bf_idx + 1:]
    return float32(new_bits), bf_idx
