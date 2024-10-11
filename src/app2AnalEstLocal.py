import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import statsmodels.api as sm
import joblib
import plotly.express as px


# Função para carregar os dados
def load_data(file_path, delimiter=';'):
    """Carrega os dados do arquivo especificado."""
    return pd.read_csv(file_path, delimiter=delimiter)


# Função para converter variáveis para numéricas
def convert_to_numeric(df, columns):
    """Converte as colunas especificadas para numéricas."""
    for coluna in columns:
        df[coluna] = pd.to_numeric(df[coluna], errors='coerce')


# Função para renomear colunas
def rename_columns(df, column_mapping):
    """Renomeia as colunas do DataFrame."""
    df.rename(columns=column_mapping, inplace=True)


# Função para remover outliers usando IQR
def remove_outliers_iqr(df, columns, multiplier=1.5):
    """Remove outliers das colunas usando o método IQR."""
    for coluna in columns:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df[coluna] >= Q1 - multiplier * IQR) & (df[coluna] <= Q3 + multiplier * IQR)
        df = df.loc[filtro]
    return df


# Função para normalizar variáveis
def normalize_variables(df, columns, scaler):
    """Normaliza as variáveis especificadas usando o StandardScaler."""
    df_normalized = df.copy()
    df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
    return df_normalized


# Função para aplicar transformação logarítmica
def apply_log_transform(df, column):
    """Aplica a transformação logarítmica à coluna especificada."""
    df[column] = np.log1p(df[column])
    return df


# Função para treinar o modelo de Regressão Linear
def train_linear_regression_model(X, y):
    """Treina um modelo de Regressão Linear."""
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


# Função para calcular o VIF
def calculate_vif(X):
    """Calcula o VIF para as variáveis."""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [
        sm.OLS(X[col].values, sm.add_constant(X.drop(col, axis=1))).fit().rsquared
        for col in X.columns
    ]
    return vif_data


# Função para treinar e avaliar um modelo (com gráficos)
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Treina um modelo de aprendizado de máquina, avalia seu desempenho e gera gráficos."""
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f'R² do modelo {model_name} nos dados de teste: {score:.2f}')

    # Previsões para o conjunto de teste
    y_pred = model.predict(X_test)

    # Criar um DataFrame para os resultados
    df_resultados = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})

    # --- Gráfico de Valores Reais vs. Previstos ---
    fig = px.scatter(df_resultados, x='Real', y='Previsto',
                     title=f'Valores Reais vs. Previstos - {model_name}',
                     trendline='ols')  # Adicionar linha de tendência
    st.plotly_chart(fig)

    # --- Gráfico de Resíduos ---
    residuals = y_test - y_pred
    fig_residuos = px.scatter(x=y_pred, y=residuals,
                              title=f'Gráfico de Resíduos - {model_name}')
    fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuos)

    return model


# Carregar os dados
df = load_data('../data/raw/dadosestacaolocal.csv')

# Converter variáveis para numéricas
colunas_para_converter = ['Outdoor Temperature(°C)', 'Outdoor Humidity(%)', 'Wind Speed(km/h)',
                          'Gust(km/h)', 'DewPoint(°C)', 'WindChill(°C)']
convert_to_numeric(df, colunas_para_converter)

# Renomear as colunas
rename_columns(
    df,
    {
        'n': 'id',
        'Time': 'date',
        'Interval': 'intervalo',
        'Indoor Temperature(°C)': 'internal_temp',
        'Indoor Humidity(%)': 'internal_humidity',
        'Outdoor Temperature(°C)': 'external_temp',
        'Outdoor Humidity(%)': 'external_humidity',
        'Relative Pressure(mmHg)': 'relative_pressure',
        'Absolute Pressure(mmHg)': 'absolute_pressure',
        'Wind Speed(km/h)': 'wind_speed',
        'Gust(km/h)': 'gust_wind',
        'Wind Direction': 'wind_direction',
        'DewPoint(°C)': 'dew_point',
        'WindChill(°C)': 'thermal_sensation',
        'Hour Rainfall(mm)': 'rain_time',
        '24 Hour Rainfall(mm)': 'rain_24h',
        'Week Rainfall(mm)': 'rain_week',
        'Month Rainfall(mm)': 'rain_month',
        'Total Rainfall(mm)': 'total_rain'
    }
)

# Remover outliers
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
df_clean = remove_outliers_iqr(df, colunas_numericas)

# Remover valores nulos
df_clean.dropna(inplace=True)

# Selecionar colunas para análise
colunas_analise = ['external_temp', 'external_humidity', 'wind_speed', 'gust_wind',
                   'dew_point', 'thermal_sensation', 'absolute_pressure']

# Título da aplicação Streamlit
st.title("Estação Meteorológica IFPR-Campus Capanema - Estudo de Caso 2")

# Criar DataFrame de correlação
corr = df_clean[colunas_analise].corr()

# Plotar o gráfico de correlação
st.write("Gráfico de correlação dos dados sem outliers")
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Verificar a multicolinearidade usando o VIF
X = df_clean[colunas_analise]
vif_data = calculate_vif(X)

# Remover variáveis com alto VIF (VIF > 5)
high_vif_variables = vif_data[vif_data['VIF'] > 5]['Variable'].tolist()
df_clean = df_clean.drop(columns=high_vif_variables)

# Plotar o gráfico de barras para o VIF
st.write("Gráfico de VIF para cada variável:")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Variable', y='VIF', data=vif_data, palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig)

# Selecionar colunas para análise após remover as variáveis com alto VIF
colunas_analise = ['external_temp', 'absolute_pressure']

# --- Modelagem ---
# Dividir os dados para treino e teste
X = df_clean[colunas_analise]
y = df_clean['external_temp']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Regressão Linear Simples ---
st.header("Regressão Linear Simples")
# Treinar e avaliar o modelo (com gráficos)
reg_simples = train_and_evaluate_model(
    LinearRegression(), X_train, X_test, y_train, y_test, "Regressão Linear Simples"
)

# --- Regressão Linear Múltipla ---
st.header("Regressão Linear Múltipla")
# Selecionar colunas para análise
colunas_analise_multipla = [
    'external_humidity',
    'wind_speed',
    'gust_wind',
    'dew_point',
    'thermal_sensation',
    'absolute_pressure',
]
# Dividir os dados para treino e teste
X_multipla = df_clean[colunas_analise_multipla]
y_multipla = df_clean['external_temp']
X_multipla_train, X_multipla_test, y_multipla_train, y_multipla_test = train_test_split(
    X_multipla, y_multipla, test_size=0.2, random_state=42
)
# Normalizar os dados
scaler_multipla = StandardScaler()
X_multipla_train_normalized = scaler_multipla.fit_transform(X_multipla_train)
X_multipla_test_normalized = scaler_multipla.transform(X_multipla_test)
# Treinar e avaliar o modelo (com gráficos)
reg_multipla = train_and_evaluate_model(
    LinearRegression(),
    X_multipla_train_normalized,
    X_multipla_test_normalized,
    y_multipla_train,
    y_multipla_test,
    "Regressão Linear Múltipla",
)
# Validação cruzada
cv_scores_multipla = cross_val_score(
    reg_multipla, X_multipla_train_normalized, y_multipla_train, cv=5
)
st.write(
    f"Acurácia média da validação cruzada (Regressão Linear Múltipla): {np.mean(cv_scores_multipla):.2f}"
)

# --- Rede Neural ---
st.header("Rede Neural")
# Normalizar os dados
scaler_nn = StandardScaler()
X_train_nn_normalized = scaler_nn.fit_transform(X_train)
X_test_nn_normalized = scaler_nn.transform(X_test)
# Treinar e avaliar o modelo (com gráficos)
reg_nn = train_and_evaluate_model(
    MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    X_train_nn_normalized,
    X_test_nn_normalized,
    y_train,
    y_test,
    "Rede Neural",
)

# --- Rede Neural Profunda ---
st.header("Rede Neural Profunda")
# Normalizar os dados
scaler_dnn = StandardScaler()
X_train_dnn_normalized = scaler_dnn.fit_transform(X_train)
X_test_dnn_normalized = scaler_dnn.transform(X_test)
# Treinar e avaliar o modelo (com gráficos)
dnn = train_and_evaluate_model(
    MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    X_train_dnn_normalized,
    X_test_dnn_normalized,
    y_train,
    y_test,
    "Rede Neural Profunda",
)