import numpy as np
import pandas as pd

def transform(raw_data_path):
    train = pd.read_csv(raw_data_path + '/train.csv')
    test = pd.read_csv(raw_data_path + '/test.csv')
    gender_submission = pd.read_csv(raw_data_path + '/gender_submission.csv')

    data = pd.concat([train, test], sort=False)

    data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)

    data['Embarked'].fillna(('S'), inplace=True)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    data['Fare'].fillna(np.mean(data['Fare']), inplace=True)

    age_avg = data['Age'].mean()
    age_std = data['Age'].std()
    data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

    delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
    data.drop(delete_columns, axis=1, inplace=True)

    train = data[:len(train)]
    test = data[len(train):]

    y_train = train['Survived']
    X_train = train.drop('Survived', axis=1)
    X_test = test.drop('Survived', axis=1)

    return y_train, X_train, X_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transform')
    parser.add_argument('--raw_data_path', type=str)
    parser.add_argument('--transformed_data_path', type=str)

    args = parser.parse_args()
    y_train, X_train, X_test = transform(**vars(args))

    y_train.to_csv(transformed_data_path + "/y_train.csv")
    X_train.to_csv(transformed_data_path + "/X_train.csv")
    X_test.to_csv(transformed_data_path + "/X_test.csv")