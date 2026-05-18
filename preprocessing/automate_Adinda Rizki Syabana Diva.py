import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing():
    data = pd.read_csv('../Titanic-Dataset.csv')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'], 
        axis=1, 
        inplace=True
    )

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    data['Sex'] = data['Sex'].map({
        'male': 0,
        'female': 1
    })

    le = LabelEncoder()
    data['Embarked'] = le.fit_transform(data['Embarked'])

    scaler = StandardScaler()
    cols = ['Age', 'Fare', 'SibSp', 'Parch']
    data[cols] = scaler.fit_transform(data[cols])

    data.to_csv(
        "Titanic-Dataset_preprocessing.csv",
        index=False
    )

    print("Preprocessing selesai.")

if __name__ == "__main__":
    preprocessing()