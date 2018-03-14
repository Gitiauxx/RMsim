from solver import SolveEquilibrium
import yaml
import numpy as np
import pandas as pd
import os
from utilities import to_change

_USGROWTH = 0.006
_yearBASE = 2015

class Forecast(object):

    def __init__(self, countyData,  commuteMatrix, tradeMatrix, popControl,
                 optParameter={'_LEARNING_RATE': 0.4, '_GTOL': 1}, counties=None, yearForecast=2040):

        """
        Solve general equilibrium until predicted values equal calibrated ones
        and there is not significant changes both wages W and prices composite PC

        :param commuteMatrix: squared matrix with the share of commuters flow from each
        (origin, destination) pair
               tradeMatrix: squared matrix with the share of commuters flow from each
        (origin, destination) pair
               popControl: labor force control forecast that the model needs to reproduce
               relative to base year
               optimization parameters: dictionary with parameter controlling the gradient descent
        """
        self._ncounty = len(countyData)
        self.commuteMatrix = commuteMatrix
        self.tradeMatrix = tradeMatrix
        self.countyData = countyData
        self.popControl = popControl
        self.PC = np.ones(self.commuteMatrix.shape[0])
        self.W = np.ones(self.commuteMatrix.shape[0])

        self.POP = np.sum(self.countyMatrix[:, 2])

        self._LEARNING_RATE = optParameter['_LEARNING_RATE']
        self._GTOL= optParameter['_GTOL']

        self.counties = counties
        self.yearForecast = yearForecast


    @classmethod
    def from_config(cls, yaml_str=None, str_or_buffer=None):

        """
        create an input configuration from a saved yaml file
        arameters
        ----------
        yaml_str : str, optional
            A YAML string from which to load model.
        str_or_buffer : str or file like, optional
            File name or buffer from which to load YAML.

        :return:
        a Forecast model
        """

        if not yaml_str and not str_or_buffer:
            raise ValueError('One of yaml_str or str_or_buffer is required.')

        if yaml_str:
            cfg = yaml.load(yaml_str)
        elif isinstance(str_or_buffer, str):
            with open(str_or_buffer) as f:
                cfg = yaml.load(f)
        else:
            cfg = yaml.load(str_or_buffer)

        countyData = pd.read_csv(os.path.join(cfg['data_directory'], cfg['county_data_filename']),
                                 index_col='county_d')
        countyData = countyData[['wages', 'emp', 'lf']]
        tradeMatrix = np.load(os.path.join(cfg['data_directory'], cfg['trade_matrix_filename']))
        commuteMatrix = np.load(os.path.join(cfg['data_directory'], cfg['commute_matrix_filename']))
        popControl = pd.read_csv(os.path.join(cfg['data_directory'], cfg['dola_forecast']))

        mod = cls(
            countyData,
            commuteMatrix,
            tradeMatrix,
            popControl,
            cfg['opt_param'],
            cfg['county_to_forecast'],
            cfg['year_forecast']
        )

        return mod

    @property
    def commuteMatrix(self):
        return self._commuteMatrix

    @commuteMatrix.setter
    def commuteMatrix(self, m):
        """
        some housekeeping to make sure that matrixes are properly dimensioned/normalized
        with values between 0 and 1
        :param m: commute matrix
        :return: m if sum equal one; return a normalized version otherwise
        """

        if m.shape != (self._ncounty, self._ncounty):
            raise Exception("Commute matrix attributes should have the same dimensions")
        if (m < 0).any():
            raise Exception("Commute matrix attributes should have nonnegative entries")
        if (np.isnan(m)).any():
            raise Exception("Commute matrix attributes cannot have NAN entries")
        if np.sum(m) == 1:
            self._commuteMatrix = m
        else:
            self._commuteMatrix = m / np.sum(m)

    @property
    def tradeMatrix(self):
        return self._tradeMatrix

    @tradeMatrix.setter
    def tradeMatrix(self, t):
        """
        some housekeeping to make sure that matrixes are properly dimensioned/normalized
        with values between 0 and 1
        :param m: trade matrix
        :return: m if sum equal one; return a normalized version otherwise
        """

        # if (t < 0).any():
        # raise Exception("Trade matrix attributes should have nonnegative entries")
        if (np.isnan(t)).any():
            raise Exception("Trade matrix attributes cannot have NAN entries")
        if np.sum(t) == 1:
            self._tradeMatrix = t
        else:
            self._tradeMatrix = t / np.sum(t)

    @property
    def _years(self):
        return np.arange(_yearBASE, self.yearForecast + 1)

    @property
    def countyMatrix(self):

        countyData = self.countyData
        countyData['A'] = 1

        return np.array(countyData[['wages', 'emp', 'lf', 'A']])


    def to_target(self):
        """
        take a csv file with labor force projection and transform it into a matrix of
        targets for the solver

        :parameters:

        :return: a matrix with #rows=#counties in countymatrix and #columns=number of year to forecast

        """

        control = to_change(self.popControl, _yearBASE, 'Labor_Force', 'county_id', 'year')
        control = control.loc[control.county_id.isin(self.counties), ['county_id', 'year', 'change']]
        control = control[control.year.isin(self._years)].set_index(['county_id', 'year']).unstack()
        control.columns = self._years

        target = self.countyData.loc[:, ['lf']]
        for y in self._years:
            target.loc[:, y] = target.loc[:, 'lf'] * (1 + _USGROWTH) ** (y - _yearBASE)
            target.loc[self.counties, y] = control.loc[:, y] * target.loc[:, 'lf']
        target = np.array(target.drop('lf', axis=1))


        return np.array(target)


    def __projections(self):

        # create solver
        solver = SolveEquilibrium(self.countyMatrix,
                                   self.commuteMatrix,
                                   self.tradeMatrix,
                                   optParameter={'_LEARNING_RATE': self._LEARNING_RATE,
                                                  '_GTOL': self._GTOL})
        # create labor force control
        control = self.to_target()

        # matrix with employment growth forecasts
        L = np.ones((control.shape))

        # check that matrices are properly balanced
        print("Checking that matrices are balanced")
        solver._check_balance()

        # forecast for all years in _years
        print( "Running forecast from %d to %d" %(_yearBASE, self.yearForecast))
        for i in np.arange(len(self._years)):
            solver.check_calibration(control[:, i])
            L[:, i] = solver.L

        return L

    @property
    def employment(self):
        emp = pd.DataFrame(self.__projections(), index=self.countyData.index, columns=self._years)
        return emp.loc[self.counties]


if __name__ == '__main__':

    import time

    yamlfile = 'Data\config_control_forecasts.yaml'
    f = Forecast.from_config(str_or_buffer=yamlfile)

    s = time.time()
    emp = f.employment
    print("time elapsed: {:.2f}s".format(time.time() - s))

    print(emp)


