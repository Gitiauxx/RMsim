import numpy as np
import pandas as pd
from extractingModels import LEHDExtractionYear, BEAExtraction

yamlfile = 'Data\config_data.yaml'

# extrract BEA data
BEAext = BEAExtraction.from_config(str_or_buffer=yamlfile)
BEAdata = BEAext.extract_all()

# extract LEHD data
LEHDext = LEHDExtractionYear.from_config(str_or_buffer=yamlfile)
block_flow = LEHDext.load_unzip()
flow = LEHDext.to_county()

# adjust earnings (by removing government contribution) and compute wages
BEAdata['earnings'] = BEAdata['earnings_by_place_of_work'] - BEAdata['contributions']
BEAdata['wages'] = BEAdata['earnings'] / BEAdata['employment']




