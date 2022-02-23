import numpy as np
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from kaggle.api.kaggle_api_extended import KaggleApi

def predictor(transformed_data_path, trained_model_path):
    X_test = pd.read_csv(transformed_data_path + "/X_test.csv", index_col=0)

    loaded_model = pickle.load(open(trained_model_path + 'model_titanic.sav', 'rb'))

    y_pred = loaded_model.predict(X_test)

    return y_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictor')
    parser.add_argument('--transformed_data_path', type=str)
    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--predicted_data_path', type=str)

    args = parser.parse_args()
    y_pred = trainer(**vars(args))

    # submit
    sub = pd.read_csv(raw_data_path + '/gender_submission.csv')
    sub['Survived'] = list(map(int, y_pred))
    sub.to_csv(predicted_data_path + 'submission.csv', index=False)

    api = KaggleApi()
    api.authenticate()
    api.competition_submit(file_name=predicted_data_path + 'submission.csv', message='update', competition='titanic')