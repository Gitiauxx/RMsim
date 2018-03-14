import pandas as pd

def to_change(data, yearBASE, var, location, year):

    """

    :param data: a data with a variable stored for different years and locations
    :param var: variable for which change needs to be computed
    :param yearBASE: base year (change =1)
    :return: a data with change computed relative to base year
    """

    dbase = data.loc[data[year] == yearBASE, [var, location]].rename(columns={var: var + 'BASE'})
    data = pd.merge(data, dbase.set_index(location), left_on=location, right_index=True)
    data['change'] = data[var] / data[var + 'BASE']

    return data[[year, location, 'change']]