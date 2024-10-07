import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

caminho_arquivo = 'https://raw.githubusercontent.com/sumitprakashdubey/Logistic-Regression/main/titanic_train.csv'
dataset = pd.read_csv(caminho_arquivo)

# Pré-processamento dos dados
dataset.drop(['Name', 'Ticket', 'Embarked', 'Cabin'], axis=1, inplace=True)
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset = pd.get_dummies(dataset, columns=['Sex'], drop_first=True)

# Separar variáveis independentes e dependentes, AQUI PODE ESTA O PROBLEMA DA COMPARAÇÃO COM O OUTRO CODIGO EM SKLEARN
X = dataset.drop('Survived', axis=1)
y = dataset['Survived']

#ividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#tensorflow

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

#Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Treinar o modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# resultado
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

