import json
import codecs
import os, sys
import numpy as np 
import argparse
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm
import glob
from logpowerspec import logpowspec
import pdb

# MINLENGTH = 600  # Only the first 600 frames are used

# def get_unified_feature(mat):
#     """ If number of frames > 600, only use the first 600 frames, 
#     otherwise, pad by repeating and then use the first 600 frames.
#     """
#     n_frames = mat.shape[0]


def build_from_path(wavlist, out_dir, stft_conf, num_workers=1, mode='train'):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    #if mode=='train':
    #    wavlist = wavlist[:5000] + wavlist[-5000:]
    #else:
    #    wavlist = wavlist[:1000] + wavlist[-1000:]
    for wav_path in wavlist:
        utt_idx = os.path.basename(wav_path).rstrip('.flac')
        futures.append(executor.submit(
            partial(_process_utterance, out_dir, utt_idx, wav_path, stft_conf)))
    return [future.result() for future in tqdm(futures)]

def _process_utterance(out_dir, utt_idx, wav_path, stft_conf):
    lps = logpowspec(wav_path, sr=stft_conf['sample_rate'], n_fft=stft_conf['n_fft'], 
            hop_length=stft_conf['hop_length'], win_length=stft_conf['win_length'], window=stft_conf['window'], pre_emphasis=stft_conf['pre_emphasis'])
    lps_filename = os.path.join(out_dir, utt_idx+".npy")
    np.save(lps_filename, lps.astype(np.float32), allow_pickle=False)
    return lps_filename


def preprocess(wavlist, out_dir, stft_conf, nj, mode='train'):
    os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(wavlist, out_dir, stft_conf, nj, mode)
    # write_metadata(metadata, out_dir)

# def write_metadata(metadata, out_dir):
#     with open(os.path.join(os.path.dirname(out_dir), 'train.txt'), 'w', encoding='utf-8') as f:
#         for m in metadata:
#             f.write('|'.join([str(x) for x in m]) + '\n')
#     samples = sum([m[2] for m in metadata])
#     sr = hparams.sample_rate
#     hours = samples / sr / 3600
#     print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), samples, hours))

def read_label_file(label_file, mode='train'):
    file_list = []
    file_labels = []
    with open(label_file, 'r') as in_f:
        for line in in_f.readlines():
            filename = line.split(' ')[1]
            if mode=='train':
              file_list.append(os.path.join(wavfile_dir_train, filename+'.flac'))
            else:
              file_list.append(os.path.join(wavfile_dir_dev, filename+'.flac'))
            file_labels.append(line.split(' ')[-1])
    return file_list, file_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='/scratch/sxliu/projects/asvspoof/ASVspoof2019-data/')
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='./_feats/LPS2')
    parser.add_argument('--access_type', type=str, default='PA')
    parser.add_argument('--param_json_path', type=str, default='./features/conf/stft.json')
    args = parser.parse_args()

    args.num_workers = args.num_workers if args.num_workers is not None else cpu_count()
    print("number of workers: ", args.num_workers)


    wavfile_dir_train = os.path.join(args.in_dir, args.access_type, f"ASVspoof2019_{args.access_type}_train", "flac")
    wavfile_dir_dev = os.path.join(args.in_dir, args.access_type, f"ASVspoof2019_{args.access_type}_dev", "flac")
    wavfile_dir_eval = os.path.join(args.in_dir, args.access_type, f"ASVspoof2019_{args.access_type}_eval", "flac")
    print(wavfile_dir_train)
    print(wavfile_dir_dev)
    wavfile_list_train = glob.glob(os.path.join(wavfile_dir_train, "*.flac"))
    wavfile_list_dev = glob.glob(os.path.join(wavfile_dir_dev, "*.flac"))
    wavfile_list_eval = glob.glob(os.path.join(wavfile_dir_eval, "*.flac"))
    #TODO: change to generic
    
    train_label_file = '/home/dlcmu09/Fake-Audio-Detection/LA_splits/ASVspoof2019.LA.cm.train_split.trn.txt'
    wavfile_list_train, wavfile_train_labels = read_label_file(train_label_file)
    dev_label_file = '/home/dlcmu09/Fake-Audio-Detection/LA_splits/ASVspoof2019.LA.cm.dev_split_earlystop.trl.txt'
    wavfile_list_dev, wavfile_dev_labels = read_label_file(dev_label_file, 'dev')
    #pdb.set_trace()
    print(len(wavfile_list_train))
    print(len(wavfile_list_dev))

    # extract LPS for training set 
    with codecs.open(args.param_json_path, 'r', encoding='utf-8') as f:
        stft_conf = json.load(f)
    print(stft_conf)

    print("Preprocess training data ...")
    preprocess(wavfile_list_train, args.out_dir, stft_conf, args.num_workers)

    print("Preprocess dev data ...")
    preprocess(wavfile_list_dev, args.out_dir, stft_conf, args.num_workers, 'dev')

    # print("Preprocess eval data ...")
    # preprocess(wavfile_list_eval, args.out_dir, stft_conf, args.num_workers)

    print("DONE!")
    sys.exit(0)



