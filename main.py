import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dados = pd.read_csv('files/dataset.csv', sep = ';')

# Estatisticas Descritivas
dados.describe().round()

# Matriz de correlação, -1 = neg perfeita e +1 = pos perfeita
dados.corr().round(4)

#### ==== Visualizando os dados em gráficos ==== ####

## Configurações de formatação dos gráficos
sns.set_palette('Accent')
sns.set_style('darkgrid')

# Boxplot da variavel 
ax = sns.boxplot(dados['Valor'], orient='h', width=0.3)
ax.figure.set_size_inches(20, 6)
ax.set_title('Preço dos Imóveis', fontsize = 20)
ax.set_xlabel('Reais', fontsize= 16)
plt.show()

# Distribuição de Frequência
ax = sns.distplot(dados['Valor'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize = 20)
ax.set_xlabel('Preço dos Imóveis (R$)', fontsize= 16)
plt.show() # Assimetria a direita (Ps: quanto mais simetrico melhor a precisão dos dados para o modelo)

# Dispersão entre as variáveis
# ax = sns.pairplot(dados, y_vars='Valor', x_vars = ['Area', 'Dist_Praia', 'Dist_Farmacia'], height= 5)
# ax.fig.suptitle('Dispersão entre as Variáveis', fontsize = 20, y = 1.05)
# plt.show()

ax = sns.pairplot(dados, y_vars='Valor', x_vars = ['Area', 'Dist_Praia', 'Dist_Farmacia'], height= 5, kind= 'reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize = 20, y = 1.05)
plt.show()

#### ==== Transformação de Variáveis ==== ####

# Distribuição normal

# Aplicando a transformação Logarítimica aos dados do dataset
dados['log_Valor'] = np.log(dados['Valor'])
dados['log_Area'] = np.log(dados['Area'])
dados['log_Dist_Praia'] = np.log(dados['Dist_Praia'] + 1)
dados['log_Dist_Farmacia'] = np.log(dados['Dist_Farmacia'] + 1)

# Distribuição de Frequencias da variavel dependente transformada (y)
ax = sns.distplot(dados['log_Valor'])
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize = 20)
ax.set_xlabel('log do Preço dos Imóveis', fontsize= 16)
plt.show()

# Dispersão entre as variáveis transformadas

ax = sns.pairplot(dados, y_vars='log_Valor', x_vars = ['log_Area', 'log_Dist_Praia', 'log_Dist_Farmacia'], height= 5, kind = 'reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize = 20, y = 1.05)
plt.show()

#### ==== Regressão Linear com StatsModel ==== ####

## Criando os datasets de treino e teste

# Criando Series pandas para armazenar o Preço dos Imóveis (y)
y = dados['log_Valor']

# Criano dataset de treino e teste
X = dados[['log_Area', 'log_Dist_Praia', 'log_Dist_Farmacia']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2811)

# Estimando o modelo com statsmodels
X_train_com_constante = sm.add_constant(X_train)
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst=True).fit()
modelo_statsmodels.summary() # para verificar a tabela de estatistica, Variaveis importantes: 'Prob (F)' e 'P > |T|' (Nesse caso distancia da farmacia era 0.6 e deve ser removido)

# Criando um novo conjunto de variáveis explicativas (X)
X = dados[['log_Area', 'log_Dist_Praia']]

# Criando novo dataset de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2811)

# Estimando o novo modelo com statsmodels
X_train_com_constante = sm.add_constant(X_train)
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst=True).fit()
modelo_statsmodels.summary() # para verificar a tabela de estatistica, Variaveis importantes: 'Prob (F)' e 'P > |T|'

#### ==== Regressão Linear com Scikit Learn ==== ####

# Instanciando a classe LinearRegression()
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Obtendo o coef de determinação (R²) do modelo estimado com dados de TREINO
print(f'R² = {modelo.score(X_train, y_train).round(3)}')

# Gerando previsões para os dados de TESTE (X_test) utilizando predict() do modelo
y_previsto = modelo.predict(X_test)

# Obtendo o coef de determinação (R²) para as previsões do modelo
print(f'R² = {metrics.r2_score(y_test, y_previsto).round(3)}')

## Obtendo previsões Pontuais
entrada = X_test[0:1]

# Gerando previsão pontual
modelo.predict(entrada)[0]

# Invertendo a transformação para obter a estimativa em R$
np.exp(modelo.predict(entrada)[0])

# Criando simulador simples

Area = 250
Dist_Praia = 1

entrada = [[np.log(Area), np.log(Dist_Praia + 1)]]
print(f'R$ {np.exp(modelo.predict(entrada)[0])}')

## Interpretação dos Coef Estimados
modelo.intercept_
np.exp(modelo.intercept_)
modelo.coef_

# Confirmando a ordem das vars explicativas no DF
X.columns

# Criando uma lista com os nomes das vars do modelo
index = ['Intercepto', 'log Area', 'log Distancia ate a Praia']

# index = ['Intercepto', 'Area (m²)', 'Distancia ate a Praia (km)']

# Criando DF para armazenar os coef do modelo

data = pd.DataFrame(data = np.append(modelo.intercept_, modelo.coef_), index = index, columns = ['Parâmetros'])
# print(data)

## Analise Graficas dos Resultados do Modelo

# Gerando as previsões do modelo para os dados de TREINO
y_previsto_train = modelo.predict(X_train)

# Grafico de dispersão entre o vlaor estimado e o valor real

ax = sns.scatterplot(x= y_previsto_train, y= y_train)
ax.figure.set_size_inches(20, 5)
ax.set_title('Previsão x Real', fontsize = 18)
ax.set_xlabel('Log do Preço - Previsão', fontsize = 14)
ax.set_ylabel('Log do Preço - Real', fontsize = 14)
plt.show()

# Obtendo residuo
residuo = y_train - y_previsto_train

# Plotando a dist de freq dos residuos
ax = sns.distplot(residuo)
ax.figure.set_size_inches(20, 5)
ax.set_title('Distribuição de Frequências dos Resíduos', fontsize = 18)
ax.set_xlabel('Log do Preço', fontsize = 14)
plt.show()