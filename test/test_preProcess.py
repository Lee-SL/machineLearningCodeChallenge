from unittest import TestCase
from preprocessing.PreProcess import PreProcess
import pandas as pd
from numpy import nan as NaN


class TestPreProcess(TestCase):

    def setUp(self):
        self.pre_process = PreProcess()
        self.df_test = pd.DataFrame({'col1': [3, 4, 2, 2, NaN],
                                      'datecol': ["2009-08-02 14:02:18", "2009-08-02 14:02:18",
                                                  "2009-08-02 14:02:18", "2009-08-02 14:02:18",
                                                  "2009-08-02 14:02:18"],
                                      'target': [1, 2, 2, 2, 2]})
        self.__numerical_cols = []


class TestRemoveDuplicates(TestPreProcess):

    def test_drop_duplicates(self):
        self.pre_process.remove_duplicates(self.df_test)
        self.assertEqual(len(self.df_test), 4)


class TestSplitDataset(TestPreProcess):

    def test_split_dataset(self):
        train, train_y, valid, valid_y = self.pre_process.split_dataset(self.df_test, "target")
        self.assertEqual(len(train), 4)
        self.assertEqual(len(valid), 1)
        self.assertEqual(train_y.columns, ["target"])


class TestStripYear(TestPreProcess):

    def test_strip_year(self):
        self.pre_process.strip_year(self.df_test, "datecol")
        self.assertEqual(len(self.df_test.datecol[0]), 4)


class TestConvertToFloat(TestPreProcess):

    def test_covert_to_float(self):
        self.pre_process.convert_to_float(self.df_test)
        self.assertEqual(self.df_test["datecol"].dtype, "float64")


class TestReplaceNaN(TestPreProcess):

    def test_replace_NaN(self):
        self.pre_process.replace_nan(self.df_test)
        self.assertEqual(self.df_test.isnull().values.any(), False)

