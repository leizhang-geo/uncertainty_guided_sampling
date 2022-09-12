#coding=utf-8

import sys
import os
import numpy as np
import config as cfg
from pygp import *


class ProgressBar:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
        self.progress_width = 50

    def update(self, step=None):
        self.current_step = step
        num_pass = int(self.current_step * self.progress_width / self.max_steps) + 1
        num_rest = self.progress_width - num_pass
        percent = (self.current_step+1) * 100.0 / self.max_steps
        progress_bar = '[' + '=' * (num_pass-1) + '>' + '-' * num_rest + ']'
        progress_bar += '%.1f' % percent + '%'
        if self.current_step < self.max_steps - 1:
            progress_bar += '\r'
        else:
            progress_bar += '\n'
        sys.stdout.write(progress_bar)
        sys.stdout.flush()
        if self.current_step >= self.max_steps:
            self.current_step = 0
            print()


def load_env_dataset(data_dir):
    fnames = os.listdir(data_dir)
    eds = EnvDataset()
    for fname in fnames:
        layer = EnvLayer(filename=os.path.join(data_dir, fname), data_type=DataTypeEnum.CONTINUOUS, layer_name=os.path.splitext(fname)[0])
        eds.add_layer(layer)
    # print('the boundary: xmin={}, ymin={}, xmax={}, ymax={}'.format(eds.desc.xmin, eds.desc.ymin, eds.desc.xmax, eds.desc.ymax))
    return eds


def simple_random_sampling(all_env_units, sample_size):
    samples_ids = np.random.choice(len(all_env_units), sample_size, replace=False)
    samples = [all_env_units[idx] for idx in samples_ids]
    return samples


def stratified_random_sampling(all_env_units, sample_size_per_strata, strata_layer_id=0):
    strata_types = [1, 2, 3, 4, 5]
    strata_ids_all = []
    for strata_type in strata_types:
        strata_ids = []
        for idx in range(len(all_env_units)):
            e = all_env_units[idx]
            if e.env_values[strata_layer_id] == strata_type:
                strata_ids.append(idx)
        strata_ids_all.append(strata_ids)
    samples_res = []
    for strata_ids in strata_ids_all:
        samples_ids = np.random.choice(strata_ids, sample_size_per_strata, replace=False)
        samples = [all_env_units[idx] for idx in samples_ids]
        samples_res.extend(samples)
    return samples_res


def log_sample_loc(f_log, sample):
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
    with open(f_log, 'a', encoding='utf-8') as f:
        f.write('{},{}\n'.format(sample.irow, sample.icol))


def log_samples_loc(f_log, samples):
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
    with open(f_log, 'w', encoding='utf-8') as f:
        f.write('loc_row,loc_col\n')
        for sample in samples:
            f.write('{},{}\n'.format(sample.irow, sample.icol))
