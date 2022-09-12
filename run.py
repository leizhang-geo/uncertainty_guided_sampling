#coding=utf-8

import os
import numpy as np
from pygp import *
import utils
import config as cfg
from geo_processing import GeoProcessing
from uncertainty_sampling import UncertaintySampling


def main():
    # Load environmental covariate dataset
    eds = utils.load_env_dataset(data_dir=cfg.data_dir)
    all_env_units = eds.env_units

    # Generate existing samples (using the random samples in this example)
    rand_samples = utils.simple_random_sampling(all_env_units, sample_size=3)

    # Construct class of uncertainty-guided sampling method
    unc_sampling = UncertaintySampling(eds, rand_samples, ex_id=cfg.ex_id)

    # Start sampling
    unc_sampling.sampling(exist_samples=rand_samples, max_iter=5, max_search_count=100)

    # Save the information of sample locations
    utils.log_samples_loc(cfg.get_fname_log_samples_loc(ex_id=cfg.ex_id), rand_samples)


if __name__ == '__main__':
    main()
