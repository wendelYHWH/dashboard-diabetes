import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#carregando o CSV 
df = pd.read_csv("diabetes.csv")

#para mostrar as primeiras linhas 
print(df.head())

#informaçoes gerais
print(df.info())

#estatisticas basicas
print(df.describe())

#AQUI JÁ IMPORTAMOS OS DADOS

#como estão distribuidas as classes e se está balanceado:

print(df['Outcome'].value_counts())
sns.countplot(x='Outcome', data=df)
plt.title('Distribuição das classes (0 = Não Diabético, 1 = Diabético)')
plt.show()

#comando para ver os nomes das colunas e os tipos de dados

print(df.dtypes)


#ver correlação entre as classes:

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()

#vamos chamar o numpy para substituir os zeros para NaN(dados não definidos/ausentes)

import numpy as np

#colunas que não podem ter zero

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

#substituição dos 0 por NaN

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

#veremos quantos NaN's teremos agora

print(df.isnull().sum())

#preenchimento dos NaN's com a mediana 
df[cols_with_zeros] = df[cols_with_zeros].fillna(df[cols_with_zeros].median())

#confirmação de limpeza

print(df.isnull().sum())

# Lista de colunas numéricas
features = ['Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']

# Plotar histogramas separados por classe (Outcome)
for col in features:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, hue='Outcome', kde=True, palette='Set1', bins=30)
    plt.title(f'Distribuição de {col} por Diagnóstico')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

#identificar outliers
for col in features:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Outcome', y=col, hue='Outcome', data=df, palette='Set2', legend=False)
    plt.title(f'Boxplot de {col} por Diagnóstico')
    plt.grid(True)
    plt.show()

#relação idade-glicose
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Age', y='Glucose', hue='Outcome', palette='Set1')
plt.title('Idade vs Glicose por Diagnóstico')
plt.grid(True)
plt.show()

#MODELAGEM COM REGRESSÃO LOGÍSTICA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Separar features (X) e target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Dividir em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar e treinar o modelo
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Matriz de Confusão - Regressão com Peso Balanceado")
plt.show()

