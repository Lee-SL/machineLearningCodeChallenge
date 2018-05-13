from sklearn.externals import joblib
import pandas as pd


class Export:

    @staticmethod
    def export_model(final_gb, model_export_dir):
        joblib.dump(final_gb, model_export_dir)

    @staticmethod
    def export_pred_file(df, df_pred, target, export_file_dir):
        df[target] = df_pred

        print("Please give your output file a name such as cats_pred.csv")
        while True:
            try:
                input_name = input().strip()
                df.to_csv(export_file_dir + input_name)
            except Exception as err:
                print("Export error encountered", err)
                print("Please enter name again:")
            else:
                print("\nExport Successful! File exported to output folder\n")
                break
