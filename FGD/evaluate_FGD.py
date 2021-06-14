import os

import torch

from embedding_space_evaluator import EmbeddingSpaceEvaluator
from train_AE import make_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_fgd(fgd_evaluator, gt_data, test_data):
    fgd_evaluator.reset()

    fgd_evaluator.push_real_samples(gt_data)
    fgd_evaluator.push_generated_samples(test_data)
    fgd_on_feat = fgd_evaluator.get_fgd(use_feat_space=True)
    fdg_on_raw = fgd_evaluator.get_fgd(use_feat_space=False)
    return fgd_on_feat, fdg_on_raw


def exp(n_frames):
    # AE model
    ae_path = f'output/model_checkpoint_{n_frames}.bin'
    fgd_evaluator = EmbeddingSpaceEvaluator(ae_path, n_frames, device)

    # load GT data
    gt_data = make_tensor('data/GroundTruth', n_frames).to(device)

    # load generated data
    generated_data_path = 'data'
    folders = sorted([f.path for f in os.scandir(generated_data_path) if f.is_dir() and 'Cond_' in f.path])

    print(f'----- EXP (n_frames: {n_frames}) -----')
    print('FGD on feature space and raw data space')
    for folder in folders:
        test_data = make_tensor(folder, n_frames).to(device)
        fgd_on_feat, fgd_on_raw = run_fgd(fgd_evaluator, gt_data, test_data)
        print(f'{os.path.basename(folder)}: {fgd_on_feat:6.3f}, {fgd_on_raw:6.3f}')
    print()


if __name__ == '__main__':
    exp(30)
    exp(60)
    exp(90)
