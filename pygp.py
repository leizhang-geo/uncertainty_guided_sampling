# coding=utf-8

from enum import Enum, unique
from osgeo import gdal
import numpy as np


@unique
class DataTypeEnum(Enum):
    CATEGORICAL = 1
    CONTINUOUS = 2


class Description:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.xmin = 0
        self.ymin = 0
        self.xmax = 999999
        self.ymax = 999999
        self.cellSize = 10
        self.no_data_value = -9999


class EnvDataset:
    def __init__(self):
        self.desc = Description()
        self.layers = []
        self.env_units = []

    def add_layer(self, new_layer):
        self.layers.append(new_layer)
        self.update_desc()
        self.update_env_units()

    def update_env_units(self):
        self.env_units = []
        if len(self.layers) <= 0:
            return
        pixel_count = self.desc.width * self.desc.height
        for i in range(pixel_count):
            irow = int(i / self.desc.width)
            icol = int(i % self.desc.width)
            e = EnvUnit()
            e.irow = irow
            e.icol = icol
            e.is_cal = True
            for j in range(len(self.layers)):
                layer = self.layers[j]
                env_value = layer.env_data[i]
                e.env_values.append(env_value)
                e.data_types.append(layer.data_type)
                if env_value == self.layers[j].no_data_value:
                    e.is_cal = False
            self.env_units.append(e)

    def update_desc(self):
        if len(self.layers) <= 0:
            return
        else:
            gdal_ds = self.layers[0].gdal_ds
            geo_transform = gdal_ds.GetGeoTransform()
            self.desc.no_data_value = gdal_ds.GetRasterBand(1).GetNoDataValue()
            self.desc.width = gdal_ds.RasterXSize
            self.desc.height = gdal_ds.RasterYSize
            self.desc.cellSize = geo_transform[1]
            self.desc.xmin = geo_transform[0]
            self.desc.ymin = geo_transform[3] - self.desc.cellSize * self.desc.height
            self.desc.xmax = self.desc.xmin + self.desc.cellSize * self.desc.width
            self.desc.ymax = geo_transform[3]

    def get_env_unit_by_rowcol(self, row, col):
        if row < 0 or row >= self.desc.height or col < 0 or col >= self.desc.width:
            return None
        else:
            return self.env_units[row * self.desc.width + col]

    def get_env_unit_by_xy(self, x, y):
        if x < self.desc.xmin or x > self.desc.xmax or y < self.desc.ymin or y > self.desc.ymax:
            return None
        else:
            irow = (self.desc.ymax - y) / self.desc.cellSize
            icol = (x - self.desc.xmin) / self.desc.cellSize
            return self.env_units[irow * self.desc.width + icol]


class EnvLayer:
    def __init__(self, filename, data_type=DataTypeEnum.CONTINUOUS, layer_name='no_name'):
        print('Loading data: <{}>'.format(filename))
        self.filename = filename
        self.layer_name = layer_name
        self.gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)
        self.x_size = self.gdal_ds.RasterXSize
        self.y_size = self.gdal_ds.RasterYSize
        self.no_data_value = self.gdal_ds.GetRasterBand(1).GetNoDataValue()
        self.data_type = data_type
        self.env_data = self.gdal_ds.GetRasterBand(1).ReadAsArray(0, 0, self.x_size, self.y_size, self.x_size, self.y_size)
        self.env_data = self.env_data.flatten()
        self.calc_stat()
        self.max_value = 0.0
        self.min_value = 0.0
        self.data_range = 0.0
        self.calc_stat()

    def calc_stat(self):
        ix = np.where(self.env_data != self.no_data_value)
        envdata_without_nodatavalue = self.env_data[ix]
        self.max_value = np.max(envdata_without_nodatavalue)
        self.min_value = np.min(envdata_without_nodatavalue)
        self.data_range = self.max_value - self.min_value


class EnvUnit:
    def __init__(self):
        self.irow = 0
        self.icol = 0
        self.is_cal = True
        self.target_value = 0.0
        self.env_values = []
        self.data_types = []
        self.uncertianty = 1.0
        self.uncertianty_tmp = 1.0
