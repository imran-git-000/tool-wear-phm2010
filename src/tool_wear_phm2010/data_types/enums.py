from enum import Enum


class Channel(Enum):
    FORCE_X = 'force_x'
    FORCE_Y = 'force_y'
    FORCE_Z = 'force_z'
    VIB_X = 'vib_x'
    VIB_Y = 'vib_y'
    VIB_Z = 'vib_z'
    AE_RMS = 'ae_rms'


class Aggregation(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
