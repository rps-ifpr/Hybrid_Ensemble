import streamlit as st
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados
dados_cultura = pd.read_csv('../data/raw/dados_cultura.csv')
X = dados_cultura[['estagio_cultura', 'tipo_solo', 'tipo_irrigacao', 'cultivar']]
y = dados_cultura['demanda_agua']
X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para treinar e avaliar o modelo com Grid Search
def treinar_modelo(X_train, X_test, y_train, y_test):
    param_grid = {
        'max_depth': [2, 5, 10, None],
        'min_samples_leaf': [1, 2, 5],
        'criterion': ['squared_error', 'friedman_mse', 'poisson'] # Adicione mais critérios
    }
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    modelo = grid_search.best_estimator_
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return modelo, mse

# Interface do Streamlit
st.title('Previsão de Demanda de Água para Culturas')
st.write('Exploração de diferentes parâmetros para o modelo de Árvore de Decisão.')

# Treinamento e avaliação do modelo
modelo, mse = treinar_modelo(X_train, X_test, y_train, y_test)

# Mostrar resultados
st.write(f'Erro Médio Quadrático (MSE): {mse}')

# Importância das features
importancias = modelo.feature_importances_
for nome_feature, importancia in zip(X.columns, importancias):
   st.write(f"{nome_feature}: {importancia:.4f}")

# Visualizar a Árvore de Decisão (com ajuste de tamanho)
fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(modelo, feature_names=X.columns, filled=True, ax=ax, max_depth=3)  # Limita a profundidade para visualização
st.pyplot(fig)