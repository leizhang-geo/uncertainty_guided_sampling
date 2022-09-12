#coding=utf-8

import os
import numpy as np
from pygp import *
import utils


class GeoProcessing:
    def __init__(self, env_dataset: EnvDataset):
        self.eds = env_dataset
        self.all_env_units = self.eds.env_units

    def calc_simi_single(self, v1, v2, data_type, data_range):
        simi = 0.0
        if data_type == DataTypeEnum.CATEGORICAL:
            if v1 == v2:
                simi = 1.0
            else:
                simi = 0.0
        elif data_type == DataTypeEnum.CONTINUOUS:  # Gower similarity
            simi = 1.0 - abs(v1 - v2) / data_range
        return simi

    def calc_simi(self, e1: EnvUnit, e2: EnvUnit):
        env_count = len(e1.env_values)
        simi = 1.0
        for i in range(env_count):
            data_type = e1.data_types[i]
            data_range = self.eds.layers[i].data_range
            simi_single = self.calc_simi_single(e1.env_values[i], e2.env_values[i], data_type, data_range)
            if simi_single <= 0.0:
                simi = 0.0
                break
            if simi_single < simi:
                simi = simi_single
        return simi

    def calc_uncertainty(self, samples, e):
        simi = 0.0
        for sample in samples:
            simi_tmp = self.calc_simi(sample, e)
            if simi_tmp > simi:
                simi = simi_tmp
        uncertainty = 1.0 - simi
        return uncertainty

    def update_uncertainty_all(self, samples):
        for e in self.all_env_units:
            if e.is_cal is False:
                continue
            e.uncertainty = self.calc_uncertainty(samples, e)

    def calc_uncertainty_all(self):
        unc_mean = 0.0
        count = 0.0
        for e in self.all_env_units:
            if e.is_cal is False:
                continue
            count += 1.0
            # unc_one = self.calc_uncertainty(samples, e)
            unc_one = e.uncertainty
            unc_mean += unc_one
        unc_mean = unc_mean / count
        return unc_mean


def main():
    np.random.seed(314)
    eds = utils.load_env_dataset(data_dir='./data')
    all_env_units = eds.env_units
    samples = utils.simple_random_sampling(all_env_units, sample_size=5)
    for sample in samples:
        print(sample.env_values[0])


if __name__ == '__main__':
    main()
