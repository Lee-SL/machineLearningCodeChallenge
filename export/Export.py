from sklearn.externals import joblib


class Export:

    @staticmethod
    def export_model(final_gb, model_export_dir):
        joblib.dump(final_gb, model_export_dir)

