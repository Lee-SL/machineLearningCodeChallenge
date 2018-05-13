import pandas as pd
import numpy as np
from numpy import nan as NaN
from sklearn import preprocessing


class PreProcess(object):

    def __init__(self):
        self.__numerical_cols = []

    @staticmethod
    def remove_duplicates(df):
        df.drop_duplicates(inplace=True)

    @staticmethod
    def split_dataset(df, target):
        train = df.sample(frac=0.8, random_state=2018)
        train_y = pd.DataFrame(train.pop(target))
        valid = df.drop(train.index)
        valid_y = pd.DataFrame(valid.pop(target))
        return train, train_y, valid, valid_y

    @staticmethod
    def strip_year(df, datecol):
        df[datecol] = [datetime[:4] for datetime in df[datecol]]

    def get_numerical_cols(self, df):
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                self.__numerical_cols.append(col)

    @staticmethod
    def __encode(charcode):
        r = 0
        ln = len(str(charcode))
        for i in range(ln):
            r += ord(str(charcode)[i])
        return r

    def convert_to_float(self, df):
        for feature in df.columns:
            if df[feature].dtype == "object":
                df[feature] = df[feature].apply(self.__encode)
            df[feature] = df[feature].astype(float)
        # missing values are converted to -1 and needs to be converted back to NaN again
        df.replace(-1, NaN, inplace=True)

    @staticmethod
    def replace_nan(df):
        for feature in df.columns:
            if df[feature].isnull().values.any():
                df[feature] = df[feature].fillna(df[feature].median(), axis=0)
        return df

    @staticmethod
    def __outliers(col):
        # replace outliers with 5th and 95th percentile
        # a number "a" from the vector "x" is an outlier if
        # a > median(x)+1.5*iqr(x) or a < median-1.5*iqr(x)
        # iqr: interquantile range = third interquantile - first interquantile
        return col[np.abs(col - col.median()) > 1.5 * (col.quantile(.75) - col.quantile(0.25))]

    def replace_outliers(self, df):
        # We need to use numerical columns before converting to float
        for col in self.__numerical_cols:
            # replace all outliers with median values
            out = self.__outliers(df[col])
            df[col].replace(to_replace=[out.min(), out.max()],
                            value=[np.percentile(df[col], 5), np.percentile(df[col], 95)],
                            inplace=True)

    @staticmethod
    def normalise(df):
        cols = df.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(df)
        df = pd.DataFrame(x_scaled)
        df.columns = cols
        return df


