import os

import soundfile
import torch
import numpy as np
import librosa
from shutil import copyfile
import pdb

def compute_loss(model, inputs, targets, criterion, compute_grad=False):
    '''
    Computes gradients of model with given inputs tand targets and loss function.
    Optionally backpropagates to compute gradients for weights.
    Procedure depends on whether we have one model for each source or not
    :param model: Model to train with
    :param inputs: Input mixture
    :param targets: Target sources
    :param criterion: Loss function to use (L1, L2, ..)
    :param compute_grad: Whether to compute gradients
    :return: Model outputs, Average loss over batch
    '''
    all_outputs = {}

    if model.separate:
        avg_loss = 0.0
        num_sources = 0
        for inst, output in model(inputs):
            loss = criterion(output, targets[inst])

            if compute_grad:
                loss.backward()

            avg_loss += loss.item()
            num_sources += 1

            all_outputs[inst] = output.detach().clone()

        avg_loss /= float(num_sources)
    else:
        loss = 0
        for inst, output in model(inputs):
            loss += criterion(output, targets[inst])
            all_outputs[inst] = output.detach().clone()

        if compute_grad:
            loss.backward()

        avg_loss = loss.item() / float(len(all_outputs))

    return all_outputs, avg_loss

def split_data_into_folders(output_dataset_dir, train_dataset_dir, dev_dataset_dir, train_split_file,  dev_split_file):

    train_output_dataset_dir = os.path.join(output_dataset_dir, 'train')
    dev_output_dataset_dir = os.path.join(output_dataset_dir, 'dev')

    train_output_dir_flag = os.path.exists(train_output_dataset_dir)
    dev_output_dir_flag = os.path.exists(dev_output_dataset_dir)

    os.makedirs(train_output_dataset_dir, exist_ok=True)
    os.makedirs(dev_output_dataset_dir, exist_ok=True)

    train_labels = {}
    with open(train_split_file, 'r') as in_f:
        lines = in_f.readlines()
        for line in lines:
            line = line.strip()
            line_words = line.split(' ')
            train_file = line_words[1]
            train_labels[train_file] = line_words[-1]
            src = os.path.join(train_dataset_dir, 'flac', train_file+'.flac') # try to remove hard-coding -- not super important
            dst = os.path.join(train_output_dataset_dir, train_file+'.flac')
            # TODO: change to moving files instead of copying if it takes too much space
            if not train_output_dir_flag:
                copyfile(src, dst)

    dev_labels = {}
    with open(dev_split_file, 'r') as in_f:
        lines = in_f.readlines()
        for line in lines:
            line = line.strip()
            line_words = line.split(' ')
            dev_file = line_words[1]
            dev_labels[dev_file] = line_words[-1]
            src = os.path.join(dev_dataset_dir, 'flac', dev_file+'.flac') # try to remove hard-coding -- not super important
            dst = os.path.join(dev_output_dataset_dir, dev_file+'.flac')
            # TODO: change to moving files instead of copying if it takes too much space
            if not dev_output_dir_flag:
                copyfile(src, dst)

    return train_labels, dev_labels

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load(path, sr=22050, mono=True, mode="numpy", offset=0.0, duration=None):
    y, curr_sr = librosa.load(path, sr=sr, mono=mono, res_type='kaiser_fast', offset=offset, duration=duration)

    if len(y.shape) == 1:
        # Expand channel dimension
        y = y[np.newaxis, :]

    if mode == "pytorch":
        y = torch.tensor(y)

    return y, curr_sr

def write_wav(path, audio, sr):
    soundfile.write(path, audio.T, sr, "PCM_16")

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def set_cyclic_lr(optimizer, it, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(it // cycle_length, cycles-1)
    curr_it = it - cycle_length * curr_cycle

    new_lr = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)

def resample(audio, orig_sr, new_sr, mode="numpy"):
    if orig_sr == new_sr:
        return audio

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    out = librosa.resample(audio, orig_sr, new_sr, res_type='kaiser_fast')

    if mode == "pytorch":
        out = torch.tensor(out)
    return out

class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def save_model(model, optimizer, step, path):
    if len(os.path.dirname(path)) > 0 and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    return step

def load_latest_model_from(model, optimizer, location):
    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)
    print("load model " + newest_file)
    return load_model(model, optimizer, newest_file)