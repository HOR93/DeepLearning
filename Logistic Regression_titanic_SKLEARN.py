import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

caminho_arquivo = ('https://raw.githubusercontent.com/sumitprakashdubey/Logistic-Regression/main/titanic_train.csv')
dataset = pd.read_csv(caminho_arquivo)

dataset

# A meta é prever a chance de sobrevivencia dos passageiros em questão
# primeiro vou analisar o dataset em varias formas
dataset.describe()

cores_x = ['red', 'blue']

# Criando o gráfico countplot com cores diferentes para x e y
sns.countplot(x='Survived', data=dataset, palette=cores_x)
plt.show()

sns.countplot(x='Survived', data=dataset, hue='Sex', palette=cores_x)
plt.show()

sns.displot(x='Age', data=dataset, palette=cores_x)

#agora ver quantos valores são null, no heatmap, onde esta em branco mostrar os valores null por coluna

dataset.isna().sum()

sns.heatmap(dataset.isna())

# agora ao inves de dar DROP nos NA no age, vamos preencher elas, já que são importantes na analise, mas CABIN podemos dropar, não influencia
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

# Pronto, agora AGE não tem mais nenhum NA
dataset['Age'].isna().sum()

dataset.drop('Cabin', axis=1, inplace=True)

dataset

# veja os tipos do dataset agora, aqui que alguns não precisam estar aqui, pois a meta é ver a chance de sobrevivencia, então não interessa nome, ticket number, logo, dropa elas
# pra isso então, vamos usar o getdummies do pandas, para transformar de object em numericos, mas pode ser feito manualmente também igual no exemplo abaixo

#dataset['Thal'] = dataset['Thal'].astype('category')
#dataset['Thal'] = dataset['Thal'].cat.codes


dataset.dtypes

# Sexo em numerico vai virar duas colunas novas, então pra isso vamos fazer o get_dummies e então separar isso em outra variavel, para então, juntar de novo no dataset

genero = pd.get_dummies(dataset['Sex'], drop_first=True)

# juntando e depois vizualize
dataset['Genero'] = genero

dataset

# apagar colunas inuteis, poderiamos ter feito isso no inicio, com exceção de Sex
dataset.drop(['Sex', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

dataset.tail()

# Agora com tudo feito, vamos separar as variaveis dependentes e independentes

x = dataset[['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Genero']]
y = dataset['Survived']

x

y

#Agora começa a modelagem do modelo, primeiro, importe Train_split do SKLEARN

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42)



# Agora import o Logistic_regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

# Finalize fazendo o PREDICT (resultado do que foi feito) e depois vizualize os resultados
predict = lr.predict(x_test)

from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])

from sklearn.metrics import classification_report

print(classification_report(y_test,predict))

