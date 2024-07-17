import pandas as pd


class BadEvaluation(dict):
    """
        Ordenes que tienen menos de 90% de accuracy en las predicciones de los msh
    """


    def get_upper_data():
        upper_csv = "Last_gum_epochs_1000_teeth_epochs_1000.csv"
        csv_file = f"D:/AI-Data/100k-Filtered_MSH/_eval/upper/{upper_csv}"
        dataframe = pd.read_csv(csv_file)
        dataframe.sort_values(["ACC"],axis=0, ascending=True,inplace=True,na_position='first')
        return dataframe.to_numpy()

    def get_lower_data():
        lower_csv = "Last_gum_epochs_1000_teeth_epochs_1000.csv"
        csv_file = f"D:/AI-Data/100k-Filtered_MSH/_eval/lower/{lower_csv}"
        dataframe = pd.read_csv(csv_file)
        dataframe.sort_values(["ACC"],axis=0, ascending=True,inplace=True,na_position='first')
        return dataframe.to_numpy()