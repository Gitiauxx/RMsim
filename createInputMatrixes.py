import numpy as np
import pandas as pd
from extractingModels import LEHDExtractionYear, BEAExtraction
import yaml
import os

yamlfile = 'Data\config_data.yaml'

SIGMA = 4
PSI = 0.49
LEARNING_RATE = 0.4
GTOL = 10**(-8)

class ToMatrix(object):

    def __init__(self, yamlfile, productivity_paramters, OUTFILE, year):
        self.yamlfile = yamlfile
        self._CACHE = {}
        self.productivity_paramters = productivity_paramters
        self.LEARNING_RATE = self.productivity_paramters['LEARNING_RATE']
        self.OUTFILE = OUTFILE
        self.year = year

    @classmethod
    def from_cfg(cls,  str_or_buffer):

        if isinstance(str_or_buffer, str):
            with open(str_or_buffer) as f:
                cfg = yaml.load(f)
        else:
            cfg = yaml.load(str_or_buffer)

        return cls(str_or_buffer,
                   cfg['productivity_estimation'],
                   cfg['Output filename'],
                   cfg['year'])

    @property
    def _BEAdata(self):
        if 'BEAdata' not in self._CACHE:
            BEAext = BEAExtraction.from_config(str_or_buffer=self.yamlfile)
            data = BEAext.get_data()
            data['earnings'] = data['earnings_by_place_of_work'] - data['contributions']
            data['wages'] = data['earnings'] / data['employment']
            self._CACHE['BEAdata'] = data
        else:
            data = self._CACHE['BEAdata']

        return data

    @property
    def flow(self):
        if 'flow' not in self._CACHE:
            LEHDext = LEHDExtractionYear.from_config(str_or_buffer=self.yamlfile)
            flow = LEHDext.get_data()
            BEAIndex = pd.DataFrame(index=self._BEAdata.index)
            flow = pd.merge(flow, BEAIndex,
                            left_on='county_o', right_index=True, how='inner')
            flow = pd.merge(flow, BEAIndex,
                            left_on='county_d', right_index='True', how='inner')
            self._CACHE['flow'] = flow
        else:
            flow = self._CACHE['flow']

        return flow

    def _county_data(self):

        data = self._BEAdata

        data = data.loc[list(set(self.flow.county_d))]
        data = data.reset_index().sort_values('county_d').set_index('county_d')

        data['emp'] = self.flow.groupby('county_d').S000.sum()
        data['lf'] = self.flow.groupby('county_o').S000.sum()

        data['earnings'] = data['emp'] * data['wages']

        commuteMatrix = self.commute_matrix
        commuteResidential = np.sum(commuteMatrix, axis=0)
        data['earnings'] = np.dot(commuteMatrix, np.array(data['earnings'] / commuteResidential))

        return data

    @property
    def county_data(self):
        if 'county_data' not in self._CACHE:
            data = self._county_data()
            data['A'] = self._update_productivity(data)
            self._CACHE['county_data'] = data
        else:
            data = self._CACHE['county_data']
        return data

    @property
    def commute_matrix(self):
        flow = self.flow
        flow['share'] = flow['S000'] / flow['S000'].sum()
        flow = flow.sort_values(['county_o', 'county_d']).set_index(['county_o', 'county_d'])
        commuteData = flow[['share']].unstack().fillna(0)

        commuteMatrix = np.array(commuteData)
        np.save(os.path.join(self.OUTFILE['data_directory'], self.OUTFILE['commute_matrix']
                             + str(self.year) + '.npy'), commuteMatrix)
        return commuteMatrix

    @property
    def trade_matrix(self):
        county_data = self.county_data
        distMatrix = self.distMatrix ** (PSI * (1 - SIGMA))

        MA = np.array((county_data['wages'] / county_data['A']) ** (1 - SIGMA) * county_data['emp'])
        MC = np.dot(np.transpose(distMatrix), MA)

        tradeMatrix = MA * distMatrix / MC[:, np.newaxis]
        np.save(os.path.join(self.OUTFILE['data_directory'], self.OUTFILE['trade_matrix']
                             + str(self.year) + '.npy'), tradeMatrix)

        return tradeMatrix


    @property
    def distMatrix(self):

        if 'dist' not in self._CACHE:

            dist = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Contro; Forecasts\\'
                   'Commuters Flows\\data\\Trade\\sf12010countydistancemiles.csv')
            dist = dist.rename(columns={"county1": "county_o", "county2":"county_d"})

            # internal distance... right now assume 5 mile
            inter_dist = pd.DataFrame(index=dist.groupby("county_o").size().index).reset_index()
            inter_dist.columns = ["county_o"]
            inter_dist["county_d"] = inter_dist["county_o"]
            inter_dist["mi_to_county"] = 10
            dist = pd.concat([dist, inter_dist], axis=0)

            dist = dist[dist.county_o.isin(self.flow.county_o)]
            dist = dist[dist.county_d.isin(self.flow.county_o)]

            dist = dist.sort_values(['county_o', 'county_d']).set_index(['county_o', 'county_d'])
            dist = dist.unstack()
            self._CACHE['dist'] = dist

        else:
            dist = self._CACHE['dist']

        return np.array(dist)

    @property
    def A(self):
        self.recalculate_productivity()

    def _recalculate_productivity(self, A, county_data):

        distMatrix = self.distMatrix ** (PSI * (1 - SIGMA))

        MA = (county_data['wages'] / A) ** (1 - SIGMA) * county_data['emp']
        MC = np.dot(np.transpose(distMatrix), MA)
        RHS = np.dot(distMatrix, np.divide(county_data['earnings'], MC))
        RHS = (county_data['wages'] ** (1 - SIGMA)) * county_data['emp'] * RHS

        return ((county_data['wages'] * county_data['emp']) / RHS) ** (1 / (SIGMA - 1))

    def _update_productivity(self, county_data):
        loss = np.inf
        A0 = np.ones(self.commute_matrix.shape[0])
        LEARNING_RATE = self.LEARNING_RATE
        iter = 0

        while loss > GTOL:
            adj = self._recalculate_productivity(A0, county_data)
            A = A0 - LEARNING_RATE * (A0 - adj)

            loss = np.mean((adj / A0 - 1)**2)
            A0 = A
            iter += 1

        return A0

if __name__ == '__main__':
    # distance data
    dist = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Contro; Forecasts\\'
                       'Commuters Flows\\data\\Trade\\sf12010countydistancemiles.csv')
    dist = dist.rename(columns={"county1": "county_o", "county2": "county_d"})

    # internal distance... right now assume 5 mile
    inter_dist = pd.DataFrame(index=dist.groupby("county_o").size().index).reset_index()
    inter_dist.columns = ["county_o"]
    inter_dist["county_d"] = inter_dist["county_o"]
    inter_dist["mi_to_county"] = 10
    dist = pd.concat([dist, inter_dist], axis=0)

    tm = ToMatrix.from_cfg(yamlfile)
    f = tm.flow
    d = tm.county_data
    print(d.loc[[8031, 8001, 8005, 8013, 8035, 8014]].A)
    print(tm.trade_matrix)













