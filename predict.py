from import_.Import import Import
from classifier.Classifier import Classifier
from export.Export import Export
from preprocessing.PreProcess import PreProcess


def main():

    date_column = "date of last vet visit"
    target = "age at death"
    export_file_dir = "./output/"
    export_model_dir = "./model/xgb_model.dat"

    # IMPORT
    import_ = Import()
    print("""
To predict how long cats will live (in years) please enter the file path
for the cats csv file for example: ./input/cats_pred.csv
    """)
    cats = import_.import_df("predict")
    cats_copy = cats.copy()

    # PRE-PROCESSING
    pre_process = PreProcess()
    print("Pre-processing Imported Data..")

    # process date to keep year only
    print("Processing date column to keep year only")
    pre_process.strip_year(cats, date_column)

    # Storing numerical columns in the background
    pre_process.get_numerical_cols(cats)

    # Convert all columns to float data type
    print("Convert all columns to float data type")
    pre_process.convert_to_float(cats)

    # Replace NaN values with Median
    print("Replacing all NaN values with median")
    cats = pre_process.replace_nan(cats)

    # Normalise dataset
    print("Normalising dataset")
    cats = pre_process.normalise(cats)
    print("""
    Cats dataset 
    {0}        
    """.format(cats.head()))

    # PREDICTION
    print("Prediction Starting")
    cats_pred = Classifier.predict(export_model_dir, cats)

    # EXPORTING
    print("Prediction Finished")
    Export.export_pred_file(cats_copy, cats_pred, target, export_file_dir)


if __name__ == "__main__":
    main()
