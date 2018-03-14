import requests
from bs4 import BeautifulSoup
import urllib
import yaml
import pandas as pd
import gc
import spatialpandas as spd
import zipfile
from io import BytesIO


class LEHDExtraction(object):

    def __init__(self, url):
        self.url = url


    def ftp_url(self):
        html_extracted = requests.get(self.url).text
        soup = BeautifulSoup(html_extracted, "lxml")
        list_url = soup.find_all('a')
        list_absurl = [urllib.parse.urljoin(self.url, list_url[i]['href']) for i in range(8, len(list_url))]
        return list_absurl

    @property
    def od_url(self):

        list_absurl = self.ftp_url()
        list_absurl_od = []

        for url_state in list_absurl:

            if url_state[-1] == '/':
                state = url_state[-3:-1]
                list_absurl_od.append([state, url_state + 'od/'])
        return list_absurl_od

    @property
    def yearMax(self):

        od = self.od_url[0]
        state = od[0]
        url = od[1]

        html_extracted = requests.get(url).text
        soup = BeautifulSoup(html_extracted, "lxml")
        list_url_year = soup.find_all('a')
        list_url_year_main = [u for u in list_url_year if u['href'][0:15] == state + '_od_main_JT00']

        list_year = [urllib.parse.urljoin(self.url, url['href']).split('.csv', 1)[0][-4:] for url in list_url_year_main]
        return max([int(i) for i in list_year])


class LEHDExtractionYear(LEHDExtraction):

    def __init__(self, url, OUTPUTFILE, year):
        self.url = url
        self.year = year
        self.OUTPUTFILE = OUTPUTFILE

    @property
    def year(self):
        return self.__year
    @year.setter
    def year(self, y):
        if y > self.yearMax:
            raise Exception('There is no LEHD data for year %d' %y)
        else:
            self.__year = y

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
        a LEHDExtraction class
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

        mod = cls(cfg['urlLEHD'],
                  cfg['LEHDOutput'],
                  cfg['year'])

        return mod

    def load_unzip(self):

        list_url = self.od_url
        data_list = []
        for u in list_url:
            state = u[0]
            if state not in ['pr', 'us', 'vi']:
                url = ''.join([u[1], state, '_od_main_JT00_', str(self.year), '.csv.gz'])
                data = pd.read_csv(url, compression='gzip')
                data_list.append(self.to_county(data))
                gc.collect()

        return pd.concat(data_list)

    def to_county(self, d):

        d['len_o'] = d['h_geocode'].apply(lambda x: len(str(x)))
        d['h_geo_str'] = d['h_geocode'].astype(str)
        d.loc[d.len_o < 15, 'h_geo_str'] = '0' + d['h_geo_str']
        d['county_o'] = d['h_geo_str'].apply(lambda x: x[0:5])

        d['len_d'] = d['w_geocode'].apply(lambda x: len(str(x)))
        d['w_geo_str'] = d['w_geocode'].astype(str)
        d.loc[d.len_d < 15, 'w_geo_str'] = '0' + d['w_geo_str']
        d['county_d'] = d['w_geo_str'].apply(lambda x: x[0:5])

        return d.groupby(['county_o', 'county_d'])[['S000']].sum()


class BEAExtraction(object):

    def __init__(self, url, key, year, variables):
        self.url = url
        self.key = key
        self.year = year
        self.variables = variables

    def __yearMax(self):
        url = "".join([self.url,
                      '?&UserID=',
                      self.key,
                      '&method=GetParameterValuesFiltered&datasetname=RegionalIncome',
                      '&TargetParameter=Year&TableName=CA4&'])

        resp = requests.get(url).json()
        yearList = [int(i['Key']) for i in resp['BEAAPI']['Results']['ParamValue']]

        return max(yearList)

    @property
    def year(self):
        return self.__year

    @year.setter
    def year(self, y):
        if y > self.__yearMax():
            raise Exception('There is no BEA data for year %d' % y)
        else:
            self.__year = y

    def build_url(self, table, linecode):

        url = "".join([self.url,
                       '?&UserID=',
                       self.key,
                       '&method=GetData&datasetname=RegionalIncome&TableName=',
                       table,
                       '&LineCode=',
                       linecode,
                       '&GeoFIPS=COUNTY&Year=',
                       str(self.year),
                       '&ResultFormat=json&'])

        return url

    def extract_url(self, table, line):
        url = self.build_url(table, line)
        resp = requests.get(url).json()
        return pd.DataFrame(resp['BEAAPI']['Results']['Data'])

    def extract_all(self):

        data_list = []

        for v in self.variables:
            t = self.variables[v].split('-', 1)[0]
            l = self.variables[v].split('-', 1)[1]
            d = self.extract_url(t, l)
            d['variable'] = v
            d['GeoFips'] = d.GeoFips.astype('int32')
            data_list.append(d)

        return pd.concat(data_list)


if __name__ == '__main__':
    url = 'https://lehd.ces.census.gov/data/lodes/LODES7/'

    urlBEA = 'https://www.bea.gov/api/data/'
    key = '182D9A25-924D-499C-82AE-913EDCB55003'
    variables = {'earnings_by_place_of_work': 'CA4-35',
                 'employment': 'CA4-7020',
                 'contributions': 'CA4-36'}

    yamlfile = 'Data\config_data.yaml'
    ext = BEAExtraction(urlBEA, key, 2016, variables)
    #d = ext.extract_all()
    #print(d)

    tiger_url ='ftp://ftp2.census.gov/geo/tiger/TIGER2016/COUNTY/'
    print(urllib.request.urlretrieve(tiger_url+ 'tl_2016_us_county.zip'))
    zf = zipfile.ZipFile(BytesIO(urllib.request.urlretrieve(tiger_url+ 'tl_2016_us_county.zip').content))
