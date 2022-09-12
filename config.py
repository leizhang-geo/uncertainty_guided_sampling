#coding=utf-8

import os


data_dir = './data'
result_dir = './results'
ex_id = 1


def get_fname_log_samples_loc(ex_id):
    return os.path.join(result_dir, 'log_samples_loc_ex_{}.csv'.format(ex_id))


def get_fname_log_sampling_process(ex_id):
    return os.path.join(result_dir, 'log_sampling_process_ex_{}.csv'.format(ex_id))
