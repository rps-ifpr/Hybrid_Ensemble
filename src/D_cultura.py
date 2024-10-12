import streamlit as st
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregamento e Preparação dos Dados
dados_cultura = pd.read_csv('../data/raw/dados_cultura.csv')
dados_cultura['estagio_e_solo'] = dados_cultura['estagio_cultura'] + "_" + dados_cultura['tipo_solo']
X = dados_cultura[['estagio_e_solo', 'tipo_irrigacao', 'cultivar']]
X = pd.get_dummies(X, drop_first=True)
y = dados_cultura['demanda_agua']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Funções para Treinar e Avaliar os Modelos
def treinar_modelo_arvore(X_treino, X_teste, y_treino, y_teste):
    param_grid = {
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'friedman_mse', 'poisson']
    }
    arvore = DecisionTreeRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=arvore, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_treino, y_treino)
    modelo = grid_search.best_estimator_
    y_pred = modelo.predict(X_teste)
    mse = mean_squared_error(y_teste, y_pred)
    return modelo, mse, y_pred

def treinar_modelo_rf(X_treino, X_teste, y_treino, y_teste):
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(X_treino, y_treino)
    modelo_rf = grid_search_rf.best_estimator_
    y_pred_rf = modelo_rf.predict(X_teste)
    mse_rf = mean_squared_error(y_teste, y_pred_rf)
    return modelo_rf, mse_rf

# 3. Interface do Streamlit
st.title('Previsão de Demanda de Água para Culturas')

# 4. Treinamento e Avaliação dos Modelos
modelo_arvore, mse_arvore, y_pred_arvore = treinar_modelo_arvore(X_treino, X_teste, y_treino, y_teste)
modelo_rf, mse_rf = treinar_modelo_rf(X_treino, X_teste, y_treino, y_teste)

# 5. Mostrar Resultados
st.subheader('Comparação de Desempenho:')
st.write(f'Erro Médio Quadrático (MSE) da Árvore de Decisão: {mse_arvore:.4f}')
st.write(f'Erro Médio Quadrático (MSE) do Random Forest: {mse_rf:.4f}')

# Gráfico de Comparação do MSE
modelos = ['Árvore de Decisão', 'Random Forest']
mse_scores = [mse_arvore, mse_rf]
plt.figure(figsize=(8, 5))
plt.bar(modelos, mse_scores, color=['skyblue', 'lightgreen'])
plt.xlabel('Modelos')
plt.ylabel('MSE')
plt.title('Comparação do Erro Médio Quadrático (MSE)')
st.pyplot(plt)

# 6. Importância das Features (apenas Árvore de Decisão)
st.subheader('Importância das Features (Árvore de Decisão):')
importancias = modelo_arvore.feature_importances_
for nome_feature, importancia in zip(X.columns, importancias):
   st.write(f"{nome_feature}: {importancia:.4f}")

# 7. Visualização da Árvore de Decisão
st.subheader('Visualização da Árvore de Decisão:')
plt.figure(figsize=(20, 10))
plot_tree(modelo_arvore, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
st.pyplot(plt)

# 8. Análise dos Resultados
st.subheader("Análise dos Resultados:")
atributo_mais_importante = X.columns[modelo_arvore.feature_importances_.argmax()]
st.write(f"**Atributo mais importante:** {atributo_mais_importante}")

