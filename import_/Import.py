import pandas as pd


class Import(object):
    def __init__(self):
        self.train_list = ['age at death', 'breed', 'date of last vet visit',
                           'hair length', 'height', 'number of vet visits', 'weight']
        self.predict_list = ['breed', 'date of last vet visit',
                             'hair length', 'height', 'number of vet visits', 'weight']

    def __check_df(self, df, type_):
        # check csv schema must contain columns type, theme, text
        if df is None:
            raise Exception("Please import csv file")

        if type_ == "train":
            for col in df.columns:
                if col in self.train_list:
                    self.train_list.remove(col)
            if len(self.train_list) != 0:
                raise Exception(
                        "{0} columns is not in the imported file please load again".format(self.train_list))
            self.train_list = ['age at death', 'breed', 'date of last vet visit',
                               'hair length', 'height', 'number of vet visits', 'weight']
        elif type_ == "predict":
            for col in df.columns:
                if col in self.predict_list:
                    self.predict_list.remove(col)
            if len(self.predict_list) != 0:
                raise Exception(
                        "{0} columns is not in the imported file please load again".format(self.predict_list))
            self.predict_list = ['breed', 'date of last vet visit',
                                 'hair length', 'height', 'number of vet visits', 'weight']

    def import_df(self, type_):
        while True:
            try:
                input_file_path = input().strip()
                df = pd.read_csv(input_file_path)
                self.__check_df(df, type_)
                if type_ == "train":
                    df = df.loc[:, self.train_list]
                elif type_ == "predict":
                    df = df.loc[:, self.predict_list]

            except Exception as err:
                print("Import error encountered", err)
                print("Please try again:")
            else:
                print("\nImport Successful!\n")
                return df

