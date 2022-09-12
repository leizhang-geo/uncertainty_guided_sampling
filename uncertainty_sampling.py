#coding=utf-8

import os
import numpy as np
import copy
import config as cfg
from pygp import *
import utils
from geo_processing import GeoProcessing


class UncertaintySampling(GeoProcessing):
    def __init__(self, eds: EnvDataset, exist_samples=None, ex_id=1):
        GeoProcessing.__init__(self, eds)
        self.ex_id = str(ex_id)
        self.samples = exist_samples
        self.unc_thred_init = 0.3
        self.unc_thred_min = 0.1
        self.o1 = 1.0
        self.o2 = 1.0
        self.o = 1.0
        self.w1 = 1.0
        self.w2 = 0.0
        self.unc_thred = self.unc_thred_init
        self.p = 1.0
        self.a = 1.0

    def calc_o1(self):
        unpredictable_prop = 0.0
        count = 0.0
        for e in self.all_env_units:
            if e.is_cal is False:
                continue
            count += 1.0
            # unc = self.calc_uncertainty(samples, e)
            unc = e.uncertainty
            if unc > self.unc_thred:
                unpredictable_prop += 1.0
        unpredictable_prop = unpredictable_prop / count
        self.o1 = unpredictable_prop
        return self.o1

    def calc_o2(self):
        self.o2 = self.calc_uncertainty_all()
        return self.o2

    def update_weights(self):
        self.w1 = np.power(self.o1, self.p)
        self.w2 = 1.0 - self.w1
    
    def update_unc_thred(self):
        if self.o2 > self.unc_thred_init:
            self.unc_thred = self.unc_thred_init
        else:
            self.unc_thred = self.unc_thred_init * np.exp(-1.0 * self.a * (self.unc_thred_init - self.o2))

    def calc_o(self):
        self.o = self.w1 * self.o1 + self.w2 * self.o2
        return self.o

    def calc_o_by_new_sample(self, new_sample):
        unc_mean = 0.0
        count = 0.0
        unpredictable_prop = 0.0
        for e in self.all_env_units:
            if e.is_cal is False:
                continue
            count += 1.0
            unc_tmp = 1 - self.calc_simi(e, new_sample)
            if unc_tmp < e.uncertainty:
                unc_mean += unc_tmp
                if unc_tmp > self.unc_thred:
                    unpredictable_prop += 1.0
            else:
                unc_mean += e.uncertainty
                if e.uncertainty > self.unc_thred:
                    unpredictable_prop += 1.0
        unpredictable_prop = unpredictable_prop / count
        unc_mean = unc_mean / count
        o1 = unpredictable_prop
        o2 = unc_mean
        o = self.w1 * o1 + self.w2 * o2
        return o

    def sampling(self, exist_samples=None, max_iter=50, max_search_count=None, rand_seed=None):
        """The core function for the uncertainty-guided sampling method.
        Args:
            exist_samples: The existing samples, it should be a list of instances of the EnvUnit class. The default value is None, so it allows no existing sample to start the sampling.
            max_iter: The maximal number of iterations. The stepwise uncertainty-guided sampling would find supplemental samples with a number equals to this parameter.
            max_search_count: For reducing the computation time if the input covariate map contains a large number of pixels, this parameter can be set as the max number of pixels as the candidate locations for searching the sample locations.
            rand_seed: The seed used by the random number generator. Pass an int for reproducible output across multiple function calls.
        """
        np.random.seed(rand_seed)
        if not os.path.exists(cfg.result_dir):
            os.mkdir(cfg.result_dir)
        with open(cfg.get_fname_log_samples_loc(self.ex_id), 'w', encoding='utf-8') as f:
            f.write('loc_row,loc_col\n')
        with open(cfg.get_fname_log_sampling_process(self.ex_id), 'w', encoding='utf-8') as f:
            f.write('iter,O1,O2,O,w1,w2,unc_thred\n')
        self.samples = exist_samples
        for iter in range(1, max_iter+1):
            print('iteration: {}'.format(iter))
            self.update_uncertainty_all(self.samples)
            self.calc_o1()
            self.calc_o2()
            self.update_unc_thred()
            self.update_weights()
            best_o = np.inf
            best_new_sample = self.all_env_units[0]
            progress_bar = utils.ProgressBar(len(self.all_env_units))
            if max_search_count is None:
                rand_idx_list = np.arange(0, len(self.all_env_units), 1)
            else:
                rand_idx_list = np.random.choice(len(self.all_env_units), max_search_count, replace=False)
                rand_idx_list = np.sort(rand_idx_list)
            for i in rand_idx_list:
                progress_bar.update(i)
                new_sample_tmp = self.all_env_units[i]
                if new_sample_tmp.is_cal is False or new_sample_tmp in self.samples:
                    continue
                o_tmp = self.calc_o_by_new_sample(new_sample_tmp)
                # print(o_tmp)
                if o_tmp < best_o:
                    best_o = o_tmp
                    best_new_sample = new_sample_tmp
            progress_bar.update(len(self.all_env_units))
            self.samples.append(best_new_sample)
            self.show_info(iter)
            utils.log_sample_loc(cfg.get_fname_log_samples_loc(self.ex_id), best_new_sample)
            self.log_info(iter)

    def show_info(self, iter):
        self.calc_o1()
        self.calc_o2()
        self.calc_o()
        print('\n-----------\n\
              iter: {}\nO1 = {:.3f}\nO2 = {:.3f}\nO = {:.3f}\nw1 = {:.3f}\nw2 = {:.3f}\nunc_thred = {:.3f}\
              \n-----------\n'.format(iter, self.o1, self.o2, self.o, self.w1, self.w2, self.unc_thred))

    def log_info(self, iter):
        with open(cfg.get_fname_log_sampling_process(self.ex_id), 'a', encoding='utf-8') as f:
            f.write('{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'
                    .format(iter, self.o1, self.o2, self.o, self.w1, self.w2, self.unc_thred))


def main():
    eds = utils.load_env_dataset(data_dir='./data')
    all_env_units = eds.env_units
    for ex_id in range(1, 10+1):
        cfg.ex_id = ex_id
        np.random.seed(ex_id)
        rand_samples = utils.simple_random_sampling(all_env_units, sample_size=5)
        rand_samples = utils.stratified_random_sampling(all_env_units, sample_size_per_strata=1)
        utils.log_samples_loc(cfg.get_fname_log_samples_loc(ex_id), rand_samples)
        unc_sampling = UncertaintySampling(eds, rand_samples, ex_id=ex_id)
        unc_sampling.sampling(exist_samples=rand_samples, max_iter=30)


if __name__ == '__main__':
    main()
