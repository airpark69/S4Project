import preprocess


folder = "your_data_folder"
preprocess.preprocessing(folder)
preprocess.make_data_yaml()
preprocess.holdout_split()