from import_.Import import Import
from preprocessing.PreProcess import PreProcess
from classifier.Classifier import Classifier
from export.Export import Export


def main():

    target = "age at death"
    date_column = "date of last vet visit"
    export_model_dir = "./model/xgb_model.dat"

    # IMPORT
    import_ = Import()
    print("""
To train model for predicting how long cats will live (in years) please enter the file path
for the cats csv file for example: ./input/cats.csv
    """)
    cats = import_.import_df()

    # PRE-PROCESSING
    pre_process = PreProcess()
    print("Pre-processing Imported Data..")
    # remove duplicates
    print("Removing Duplicates")
    pre_process.remove_duplicates(cats)

    # Split into training and validation data set
    print("Splitting data into training and validation set")
    train, train_y, valid, valid_y = pre_process.split_dataset(cats, target)

    # process date to keep year only
    print("Processing date column to keep year only")
    pre_process.strip_year(train, date_column)
    pre_process.strip_year(valid, date_column)

    # Storing numerical columns in the background
    pre_process.get_numerical_cols(train)

    # Convert all columns to float data type
    print("Convert all columns to float data type")
    pre_process.convert_to_float(train)
    pre_process.convert_to_float(valid)

    # Replace NaN values with Median
    print("Replacing all NaN values with median")
    train = pre_process.replace_nan(train)
    train_y = pre_process.replace_nan(train_y)
    valid = pre_process.replace_nan(valid)
    valid_y = pre_process.replace_nan(valid_y)

    # Replace Outliers
    print("Replacing outliers")
    pre_process.replace_outliers(train)
    pre_process.replace_outliers(valid)

    # Normalise dataset
    print("Normalising dataset")
    train = pre_process.normalise(train)
    valid = pre_process.normalise(valid)
    print("""
Train dataset 
{0}        
""".format(train.head()))
    print("""
Valid dataset 
{0}        
""".format(valid.head()))

    # MODELLING
    print("Modelling Starting")
    clf = Classifier(train, train_y)

    print("Cross Validation Starting")
    cv = clf.cross_validate()

    # Best number of trees
    best_num = cv.shape[0]
    print("Best number of trees: {}".format(best_num))

    # Cross Validation mean absolute error
    cv_mae = cv.iloc[best_num-1, 0]

    print("Final Model Plotting using best number of trees")
    final_gb = clf.train_model(best_num)

    # Exporting Model
    print("Exporting Model to model folder")
    Export.export_model(final_gb, export_model_dir)

    # Validation
    print("Validating Model")
    valid_pred = clf.predict(export_model_dir, valid)
    valid_mae = clf.mae(valid_y, valid_pred)

    print("""
Model Validated Final Stats

Cross-Validation Mean Absolute Error: {0:.2f} years
Validation Mean Absolute Error: {1:.2f} years
""".format(cv_mae, valid_mae))


if __name__ == "__main__":
    main()
