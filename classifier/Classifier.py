import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error


class Classifier(object):

    def __init__(self, train, train_y):
        self.xgb_params = {
            'seed': 0,
            'colsample_bytree': 0.8,
            'silent': 1,
            'subsample': 0.6,
            'learning_rate': 0.01,
            'objective': 'reg:linear',
            'max_depth': 4,
            'num_parallel_tree': 1,
            'min_child_weight': 1,
            'eval_metric': 'mae'
        }
        self.xgdmat = xgb.DMatrix(train, train_y)

    def cross_validate(self):
        return xgb.cv(params=self.xgb_params, dtrain=self.xgdmat,
                      num_boost_round=347, nfold=10, stratified=False,
                      early_stopping_rounds=50, show_stdv=True)

    def train_model(self, best_num):
        return xgb.train(self.xgb_params, self.xgdmat, num_boost_round=best_num)

    @staticmethod
    def predict(model_location, data):
        clf = joblib.load(model_location)
        data_mat = xgb.DMatrix(data)
        pred = clf.predict(data_mat)
        return pred

    @staticmethod
    def mae(target, pred):
        return mean_absolute_error(target, pred)
