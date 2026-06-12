# config.py
from enum import Enum
import rapidwright

from com.xilinx.rapidwright.device import SiteTypeEnum

class PlaceType(Enum):
    CENTERED = 1
    IO = 2
    OTHER = 3

class GridType(Enum):
    SQUARE = 1
    RECTAN = 2
    OTHER = 3

class IoMode(Enum):
    NORMAL = 1
    VIRTUAL_NODE = 2

SLICE_SITE_ENUM = [SiteTypeEnum.SLICEL, SiteTypeEnum.SLICEM]

DSP_SITE_ENUM = [SiteTypeEnum.DSP48E2]

IO_SITE_ENUM = [
    SiteTypeEnum.HPIOB,
    SiteTypeEnum.HRIO
]

# CLOCK_SITE_ENUM = [SiteTypeEnum.BUFGCE]

OTHER_SITE_ENUM = [SiteTypeEnum.BUFGCE, 
                   SiteTypeEnum.RAMB36, 
                   SiteTypeEnum.RAMB180, 
                   SiteTypeEnum.BITSLICE_COMPONENT_RX_TX,                   
                   SiteTypeEnum.HPIOB_S,
                   SiteTypeEnum.HPIOB_M,
                   SiteTypeEnum.HPIOB_SNGL,
                   SiteTypeEnum.HDIOLOGIC_M,
                   SiteTypeEnum.HDIOLOGIC_S,
                   SiteTypeEnum.HDIOB_M,
                   SiteTypeEnum.HDIOB_S]