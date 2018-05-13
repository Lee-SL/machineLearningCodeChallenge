import pandas as pd


class Import(object):
    def __init__(self):
        self.train_list = ['age at death', 'breed', 'date of last vet visit',
                           'hair length', 'height', 'number of vet visits', 'weight']

    def __check_df(self, df):
        # check csv schema must contain columns type, theme, text
        if df is None:
            raise Exception("Please import csv file")

        for col in df.columns:
            if col in self.train_list:
                self.train_list.remove(col)
        if len(self.train_list) != 0:
            raise Exception(
                "{0} columns is not in the imported file please load again".format(self.train_list))
        self.train_list = ['age at death', 'breed', 'date of last vet visit',
                           'hair length', 'height', 'number of vet visits', 'weight']

    def import_df(self):
        while True:
            try:
                df = pd.read_csv(input().strip())
                self.__check_df(df)
                df = df.loc[:, self.train_list]
                df
            except Exception as err:
                print("Import error encountered", err)
                print("Please try again:")
            else:
                print("\nImport Successful!\n")
                return df
                break

