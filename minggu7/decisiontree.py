import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dataset Play Tennis
data = {
    'Outlook': ['Sunny','Sunny','Cloudy','Rainy','Rainy','Rainy','Cloudy','Sunny',
                'Sunny','Rainy','Sunny','Cloudy','Cloudy','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Windy': ['No','Yes','No','No','No','Yes','Yes','No',
              'No','No','Yes','Yes','No','Yes'],
    'Play': ['No','No','Yes','Yes','Yes','Yes','Yes','No',
             'Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

# Encoding
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

X = df.drop('Play', axis=1)
y = df['Play']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

