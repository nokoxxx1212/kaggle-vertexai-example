import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

def trainer(transformed_data_path):
    y_train = pd.read_csv(transformed_data_path + "/y_train.csv", index_col=0)
    X_train = pd.read_csv(transformed_data_path + "/X_train.csv", index_col=0)

    model = LogisticRegression(penalty='l2', solver='sag', random_state=0)
    model.fit(X_train, y_train)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--transformed_data_path', type=str)
    parser.add_argument('--trained_model_path', type=str)

    args = parser.parse_args()
    model = trainer(**vars(args))

    pickle.dump(model, open(trained_model_path + 'model_titanic.sav', 'wb'))