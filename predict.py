# import_ all methods here
from importClass.Import import Import
from export.Export import Export
from preprocessing.ppCalcs import *
from preprocessing.PreProcess import *


def Main():
    # initialise comments
    comment = ""
    # import_
    importOb = Import()
    print("""
To summarise survey results please enter the file path for both the survey and the survey responses
Please enter the survey file path for example: ./example-data/survey-1.csv 
    """)
    survey = importOb.importSurveys(input())

    print("""
Please enter the survey response file path for example: ./example-data/survey-1-responses.csv
    """)
    survey_res = importOb.importSurveys(input())
    importOb.checkSurveys(survey, survey_res)

    # Calculations
    # participation preprocessing
    # initialise object
    ppCalc = PPCalcs(survey_res, comment)
    ppCount, ppPercent = ppCalc.ppCalc()

    # average rating preprocessing
    # initialise object
    avgRQ = AverageRating(survey, survey_res, comment)
    avgs, comment = avgRQ.avgRating()

    # export class
    export = Export(ppCount, ppPercent, avgs, comment)
    export.displayRes()

if __name__ == "__main__":
    Main()
