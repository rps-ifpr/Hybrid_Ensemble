import matplotlib.pyplot as plt
import numpy as np

# Dados de exemplo para o gráfico
training_sizes = np.array([50, 100, 150, 200, 250, 300])  # Tamanho do conjunto de treinamento
mse_train = np.array([0.17, 0.15, 0.13, 0.12, 0.11, 0.10])  # MSE para os dados de treinamento
mse_val = np.array([0.18, 0.17, 0.16, 0.18, 0.19, 0.20])  # MSE para os dados de validação
std_train = np.array([0.02, 0.015, 0.01, 0.01, 0.01, 0.01])  # Desvio padrão do MSE para treinamento
std_val = np.array([0.025, 0.02, 0.015, 0.02, 0.03, 0.03])  # Desvio padrão do MSE para validação

# Gerando o gráfico
plt.figure(figsize=(8, 6))

# Curva para o MSE de treinamento
plt.plot(training_sizes, mse_train, 'bo-', label='Training')
plt.fill_between(training_sizes, mse_train - std_train, mse_train + std_train, color='blue', alpha=0.2)

# Curva para o MSE de validação
plt.plot(training_sizes, mse_val, 'go-', label='Validation')
plt.fill_between(training_sizes, mse_val - std_val, mse_val + std_val, color='green', alpha=0.2)

# Rótulos e título
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('Relationship between the size of the training set and the MSE')

# Exibindo a legenda
plt.legend()

# Exibindo o gráfico
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Dados de exemplo para o gráfico
training_sizes = np.array([50, 100, 150, 200, 250, 300])  # Tamanho do conjunto de treinamento
mse_train = np.array([0.18, 0.16, 0.14, 0.13, 0.12, 0.10])  # MSE para os dados de treinamento
mse_val = np.array([0.18, 0.17, 0.17, 0.18, 0.19, 0.20])  # MSE para os dados de validação
std_train = np.array([0.02, 0.015, 0.01, 0.01, 0.01, 0.01])  # Desvio padrão do MSE para treinamento
std_val = np.array([0.025, 0.02, 0.015, 0.02, 0.03, 0.03])  # Desvio padrão do MSE para validação

# Gerando o gráfico
plt.figure(figsize=(8, 6))

# Curva para o MSE de treinamento
plt.plot(training_sizes, mse_train, 'bo-', label='Training')
plt.fill_between(training_sizes, mse_train - std_train, mse_train + std_train, color='blue', alpha=0.2)

# Curva para o MSE de validação
plt.plot(training_sizes, mse_val, 'go-', label='Validation')
plt.fill_between(training_sizes, mse_val - std_val, mse_val + std_val, color='green', alpha=0.2)

# Rótulos e título
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title('Model Learning Curve')

# Exibindo a legenda
plt.legend()

# Exibindo o gráfico
plt.grid(True)
plt.show()

