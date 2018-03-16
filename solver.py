import numpy as np
import yaml
import os

_EPSILON = 4.0
_ALPHA = 0.6
_SIGMA = 4.0
_LEARNING_RATE = 0.4
_TOLCONV = 1


class SolveEquilibrium(object):

    def __init__(self, countyMatrix,  commuteMatrix, tradeMatrix, optParameter={'_LEARNING_RATE': 0.4}):

        """
        Solve general equilibrium until predicted values equal calibrated ones
        and there is not significant changes both wages W and prices composite PC

        :param commuteMatrix: squared matrix with the share of commuters flow from each
        (origin, destination) pair
               tradeMatrix: squared matrix with the share of commuters flow from each
        (origin, destination) pair
             optimization parameters: dictionary with parameter controlling the gradient descent
        """
        self._ncounty = countyMatrix.shape[0]

        self.commuteMatrix = commuteMatrix
        self.tradeMatrix = tradeMatrix
        self.countyMatrix = countyMatrix
        self.PC = np.ones(self.commuteMatrix.shape[0])
        self.W = np.ones(self.commuteMatrix.shape[0])

        self.POP = np.sum(self.countyMatrix[:, 2])

        self._LEARNING_RATE = optParameter['_LEARNING_RATE']
        self._GTOL= optParameter['_GTOL']



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
        a SolveEquilbirum class
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

        countyMatrix = pd.read_csv(os.path.join(cfg['data_directory'], cfg['county_data_filename']))
        countyMatrix['A'] = 1
        countyMatrix = np.array(countyMatrix[['wages', 'emp', 'lf', 'A', 'avg_earnings']])
        tradeMatrix = np.load(os.path.join(cfg['data_directory'], cfg['trade_matrix_filename']))
        commuteMatrix = np.load(os.path.join(cfg['data_directory'], cfg['commute_matrix_filename']))

        mod = cls(
            countyMatrix,
            commuteMatrix,
            tradeMatrix,
            cfg['opt_param']
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

        #if (t < 0).any():
            #raise Exception("Trade matrix attributes should have nonnegative entries")
        if (np.isnan(t)).any():
            raise Exception("Trade matrix attributes cannot have NAN entries")
        if (np.sum(t, axis=0) == 1).all():
            self._tradeMatrix = t
        else:
            self._tradeMatrix = t / np.sum(t, axis=0)

    @property
    def countyMatrix(self):
        return self._countyMatrix

    @countyMatrix.setter
    def countyMatrix(self, c):
        """
        some housekeeping to make sure that the county matrix does not have negative values or NAN
        :param c: county matrix
        :return: c if pass barriers
        """

        #if (c < 0).any():
            #raise Exception(" County matrix attributes should have nonnegative entries")
        if (np.isnan(c)).any():
            raise Exception("County matrix attributes cannot have NAN entries")

        self._countyMatrix = c


    def __recalculate_employment(self):
        """
        :return: employment growth using equation () from documentation
        """
        L = np.dot(self.commuteMatrix.transpose(), self.PC ** (-_EPSILON))
        L = L * (self.W ** _EPSILON)
        L = L / np.sum(L)
        L = L * self.POP / self.countyMatrix[:, 1]

        return L

    def __recalculate_market_attractivity(self):
        """
        :return: each market attractivity using equation () from documentation
        """
        return np.dot(self.commuteMatrix, self.W ** _EPSILON)

    def __recalculate_labor_force(self):
        """
        :return: labor force growth using equation () from documentation
        """
        R = self._MA() * self.PC ** (-_EPSILON)
        R = R / np.sum(R)
        return R * self.POP / self.countyMatrix[:, 2]

    def __recalculate_price(self, target):

        return self.PC * (target / self.predicted) ** (-1 / _EPSILON)

    def __recalculate_income(self):
        """
        :return: average income growth using equation () from documentation
        """
        I = np.dot(self.commuteMatrix, self.W ** (_EPSILON + 1) * self.countyMatrix[:, 0])
        return I / self._MA(in_iter=1)

    def __recalculate_market_competitivity(self):
        """
        :return: change in market competitivity for each market using equation () from documentation
        """
        return (self.W / self.countyMatrix[:, 3]) ** (1 - _SIGMA)

    def __recalculate_wages(self):

        L = self.L
        MC = self._MC
        P = np.dot(self.tradeMatrix.transpose(), L * MC)
        MP = self._R(in_iter=0) * self._I * self.countyMatrix[:, 2] / P
        return np.dot(self.tradeMatrix, MP) * MC / (self.countyMatrix[:, 1] * self.countyMatrix[:, 0])

    def __update_variables(self, target):

        self.PC = self.PC - self._LEARNING_RATE * (self.PC - self.__recalculate_price(target))
        self.W = self.W - self._LEARNING_RATE * (self.W - self.__recalculate_wages())

    def _check_balance(self):

        """
        Make sure that the initial natrixes are balanced..
        To do so, run the equations with a  and check that no values are changed
        :return: error message if the system is not balanced
        """
        self.PC[:1000] = 2
        self.check_calibration(self.countyMatrix[:, 2])

        wageError = np.sum((self.W /self.W[0] - 1)**2)
        print(wageError)
        if wageError > 0.001:
            raise Exception('The initial system is not balanced. Check that the parameter values '
                            'are consistent with input matrixes')


    def check_calibration(self, target):

        self.POP = np.sum(target)
        loss = np.inf

        while loss > self._GTOL:
            self.__update_variables(target)
            loss = self._loss(target)

    @property
    def L(self):
        """
        :return: the current employment growth
        """
        return self.__recalculate_employment()

    @property
    def R(self,):
        """
        :return: the current labor force growth
        """
        return self.__recalculate_labor_force()

    def _R(self, in_iter=0):
        if in_iter == 0:
            self.__R = self.__recalculate_labor_force()

        return self.__R

    def _MA(self, in_iter=0):
        """
        :return: the current change in market attractivity
        """
        if in_iter == 0:
            self.__MA = self.__recalculate_market_attractivity()

        return self.__MA


    @property
    def _I(self):
        """
        :return: the current growth in income
        """
        return self.__recalculate_income()

    @property
    def _MC(self):
        """
        :return: the current change in market competitivity
        """
        return self.__recalculate_market_competitivity()

    @property
    def predicted(self):
        """
        :return: the current change in market competitivity
        """
        return self._R() * self.countyMatrix[:, 2]

    def _loss(self, target):
        """
        Loss expressed as square root of the mean of
        the difference between predicted values and target values
        :return: current loss function between target and prediction
        """
        return np.sqrt(np.mean((target - self.predicted)**2))


class SolveProductivity(object):

    def __init__(self, countyMatrix, distMatrix, commuteMatrix, prodParameters={'SIGMA':4, 'PSI':0.49}):
        self.countyMatrix = countyMatrix
        self.distMatrix = distMatrix
        self.commuteMatrix = commuteMatrix

    @property
    def countyMatrix(self):
        return self._countyMatrix

    @countyMatrix.setter
    def countyMatrix(self, c):
        """
        some housekeeping to make sure that the county matrix does not have negative values or NAN
        :param c: county matrix
        :return: c if pass barriers
        """

        # if (c < 0).any():
        # raise Exception(" County matrix attributes should have nonnegative entries")
        if (np.isnan(c)).any():
            raise Exception("County matrix attributes cannot have NAN entries")

        self._countyMatrix = c

if __name__ == '__main__':


    import pandas as pd

    year=2015
    year_forecast=2040
    tar_area = [8001, 8005, 8013, 8014, 8031, 8035, 8059, 8123, 8041]

    input_directory = 'C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\Contro; Forecasts\\Commuters Flows\\data\\InputsModel'

    growth = pd.read_csv('C:\\Users\\xgitiaux\\Documents\\Research and Analysis\\'
                         'Contro; Forecasts\\Commuters Flows\\data\\DOLA\\labor_force_county_DOLA_2017.csv')
    growth['county'] = 8000 + growth.county_id

    # define target areas as all counties in Colorado:
    baseyear = growth[growth.year == year]
    baseyear = baseyear.rename(columns={'Labor_Force': 'lfbase'})
    growth = pd.merge(growth, baseyear[['county', 'lfbase']], on='county')
    growth['change'] = growth['Labor_Force'] / growth['lfbase']
    growth.to_csv(os.path.join(input_directory, 'labor_force_change_DOLA_2017.csv'))

    target = growth[growth.county.isin(tar_area)][['county', 'year', 'change']]
    target = target[(target.year >= year) & (target.year <= year_forecast)].set_index(['county', 'year']).unstack()
    target.columns = np.arange(year, year_forecast + 1)

    filename_d = os.path.join(input_directory, 'd_matrix.csv')
    d = pd.read_csv(filename_d, index_col='county_d')

    filename_flow = os.path.join(input_directory, 'flow_matrix.npy')
    fMatrix = np.load(filename_flow)

    filename_trade = os.path.join(input_directory, 'flow_trade.npy')
    piMatrix = np.load(filename_trade)

    d['A'] = 1

    # target
    tar = d[['lf']]
    for y in np.arange(year, year_forecast + 1):
        tar.loc[:, y] = tar.loc[:, 'lf'] * (1 + 0.006) ** (y - year)
        tar.loc[target.index, y] = target[y] * tar.loc[target.index, 'lf']
    target = np.array(tar.drop('lf', axis=1))


    yamlfile = 'Data\config_control_forecasts.yaml'
    se = SolveEquilibrium.from_config(str_or_buffer=yamlfile)


    import time


    s = time.time()
    se.check_calibration(target[:, 25])


    print("time elapsed: {:.2f}s".format(time.time() - s))
    d['predicted'] = se.predicted
    print(d.loc[tar_area].predicted)




