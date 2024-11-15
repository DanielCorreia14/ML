import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Leitura e limpeza dos dados
bd = pd.read_csv("ibovespa.csv", sep=";", names=["ibovespa"], decimal=".") 
bd["ibovespa"] = pd.to_numeric(bd["ibovespa"], errors="coerce")  # Converte valores inválidos para NaN
bd.dropna(inplace=True)  # Remove linhas com NaN

# Normalização dos dados
x = bd["ibovespa"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
x_normalized = scaler.fit_transform(x)

# Criação do dataset janelado com TimeseriesGenerator
tam_janela = 10
generator = TimeseriesGenerator(x_normalized, x_normalized, length=tam_janela, batch_size=1)

# Extração de X e y das janelas
X, y = [], []

for i in range(len(generator)):
    x_batch, y_batch = generator[i]
    X.append(x_batch[0, :-1])  # Características (input): pega tudo, exceto o último valor da janela
    y.append(y_batch[0, -1])   # Rótulos (output): pega o último valor da janela

X = np.array(X)
y = np.array(y)

# Verificando o formato de X e y
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

# Agora, X é de dimensão (n_amostras, n_features)
X = X.reshape(X.shape[0], -1)  # Achata qualquer dimensão extra, resultando em (n_amostras, n_features)

# Visualização do dataset janelado
plt.plot(generator[0][0][:, 0])  # Exemplo de como a janela aparece
plt.title("Primeira Janela do Dataset")
plt.savefig("primeira_janela.png")  
plt.close()  

# Separação do conjunto de treinamento e de teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Definição do modelo
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', 
                     max_iter=60000, learning_rate_init=0.001)

# Treinamento do modelo
model.fit(X_train, y_train)

# Avaliação do modelo
predictions_train = model.predict(X_train)
predictions_test = model.predict(X_test)

# Métricas de avaliação
print("MSE (Treinamento):", mean_squared_error(y_train, predictions_train))
print("MSE (Teste):", mean_squared_error(y_test, predictions_test))
print("MAE (Treinamento):", mean_absolute_error(y_train, predictions_train))
print("MAE (Teste):", mean_absolute_error(y_test, predictions_test))
print("R² (Treinamento):", r2_score(y_train, predictions_train))
print("R² (Teste):", r2_score(y_test, predictions_test))

# Curva de perda durante o treinamento
plt.plot(model.loss_curve_)
plt.title("Erro RMS durante o Treinamento")
plt.xlabel("Iterações")
plt.ylabel("Erro RMS")
plt.savefig("curva_perda.png")  
plt.close()

# Plotagem dos dados de treinamento
plt.plot(range(len(y_train)), y_train, label="Real", marker='o', linestyle='none', color='blue', markersize=4)
plt.plot(range(len(predictions_train)), predictions_train, label="Predito", marker='x', linestyle='none', color='red', markersize=4)
plt.title("Dados de Treinamento")
plt.xlabel("Amostras")
plt.ylabel("Valores Normalizados")
plt.legend()
plt.savefig("dados_treinamento.png")  
plt.close()

# Gráfico de Resíduos (erro entre previsão e valor real)
residuals = y_test - predictions_test  
plt.scatter(range(len(residuals)), residuals)
plt.hlines(y=0, xmin=0, xmax=len(residuals), color='r', linestyle='--')
plt.title("Resíduos - Diferença entre Previsões e Valores Reais")
plt.xlabel("Amostras")
plt.ylabel("Resíduos")
plt.savefig("residuos.png")  
plt.close()


# Gráfico de Resíduos (erro entre previsão e valor real)
residuals = y_test - predictions_test