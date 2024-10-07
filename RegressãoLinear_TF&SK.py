import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

caminho_arquivo = 'https://raw.githubusercontent.com/rashida048/TensorFlow-Tutorial/main/Housing.csv'
dataset = pd.read_csv(caminho_arquivo)

dataset.head() # quero o price das casas

#limpar a coluna que não vai ser usada ou que não tem relevancia
dataset = dataset.drop(columns=['Unnamed: 0'])

#procurar non values, se for = 0, então ta ok
dataset.isna().sum()

#agora criar as variaveis de teste
train, test = train_test_split(dataset, test_size=0.2, random_state=21)

# tudo menos a coluna price
train_x = train.drop(columns=['price'])
test_x = test.drop(columns=['price'])


# apenas o price
train_y = train['price']
test_y = test['price']

train_x

#fazer a normalização, pois alguns como area não estão em 1 e 0
train_stats = train_x.describe()
train_stats

# preciso apenas o STD e MEAN para normalização, então vamos fazer uma matriz transposta
train_stats = train_stats.transpose()
train_stats

#função para normalizar
def normalizer(x):
  return(x - train_stats['mean']) / train_stats['std']

train_x = normalizer(train_x)
test_x = normalizer (test_x)

train_x

# rede neural
import keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (12, )),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

from tensorflow import keras

model.compile(optimizer = keras.optimizers.Adam(learning_rate= 1e-2),
              loss = 'mean_squared_error',
              metrics = tf.keras.metrics.RootMeanSquaredError())

history = model.fit(train_x, train_y, epochs = 100)

loss = history.history['loss']

loss

#visualizar
plt.plot(range(100), loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

predictions = model.predict(test_x)

# Exibir as previsões feitas pelo modelo
print("Previsões do modelo para o preço das casas:")
print(predictions[:10])  # Mostrando as previsões para as primeiras 10 casas nos dados de teste

# Exibir os verdadeiros valores de preço das casas
print("\nValores reais de preço das casas:")
print(test_y[:10])  # Mostrando os verdadeiros valores de preço das primeiras 10 casas nos dados de teste
