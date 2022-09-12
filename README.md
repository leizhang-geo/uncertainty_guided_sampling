# Adaptive uncertainty-guided sampling (AUGSS) method

This repository contains the code used for the analysis in the paper:

Zhang et al. 2022. An adaptive uncertainty-guided sampling method for geospatial prediction and its application in digital soil mapping. *International Journal of Geographical Information Science*

## Requirement

- Python3 (>=3.7)
- Numpy
- GDAL

## Description of data

- **data (directory)**: The environmental covariate data folder.
- **config.py**: The configuration file for setting the log file path.
- **pygp.py**: For generating the evironmental layers.
- **geoprocessing.py**: The class for calculating the environmental similarity and prediction uncertainty.
- **uncertainty_sampling.py**: The core function for generating the uncertainty-guided sampling method.
- **utils.py**: It contains functions for the data loading and storing the result.
- **run.py**: The file that the program start to run.

## Usage instructions

### Preparation
All data (environmental covariate maps) need to be placed in the folder namely "data" under the root directory. You can add your data or use the raster example data that we provided.

The results of the sampling would be stored in the folder namely "results" under the root directory. This folder can be automatically created when starting the program.

The directories of data and results can also be customized in **config.py** file.

### Parameter setting
The parameters of the sampling method can be set when using the function of *sampling* in **uncertainty_sampling.py** file. The detailed explanation of parameters can be found in the description of this function.

### Running the program
```
python run.py
```
The sample locations and the process information would be saved in the result files.

### License

[MIT License](./LICENSE)

## Contact

For questions and supports please contact the author: Lei Zhang 张磊 (lei.zhang.geo@outlook.com)

Lei Zhang's [Homepage](https://leizhang-geo.github.io/)
