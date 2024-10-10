# 🚀 Hybrid Ensemble Learning for Precision Irrigation in Hop Production

This repository contains the code and documentation for my doctoral project, which proposes a **hybrid ensemble learning approach for real-time precision irrigation in hop production**.

## 🌾 Context and Motivation

Efficient hop production demands precise irrigation management to optimize crop yield and quality while minimizing water consumption and environmental impact. Precision irrigation, guided by data and predictive models, emerges as a promising solution.

## 💡 Proposed Approach

This work explores the development of a real-time decision support system for precision irrigation in hop plantations. The backbone of the system is a **hybrid ensemble learning model**, which combines the strengths of different machine learning algorithms for robust and accurate predictions.

### Key features:

- **Multi-sensor data:** Integration of climate, soil, and plant data to capture the complex system dynamics.
- **Predictive modeling:** Development of a hybrid ensemble model to predict irrigation needs with high accuracy.
- **User-friendly interface:** Implementation of an intuitive web application (Streamlit) for data visualization and interaction with the model.

## 📂 Repository Structure
```bash
/projeto/
│
├── .git/                     # Controle de versão Git
├── .venv/                    # Ambiente virtual Python para isolamento de dependências
├── .idea/                    # Configurações do IDE (IntelliJ, PyCharm, etc.)
├── .gitignore                # Arquivos a serem ignorados no Git
├── README.md                 # Documentação do projeto
├── requirements.txt          # Lista de dependências Python
├── LICENSE                   # Licença do projeto
│
├── src/                      # Códigos-fonte
│   ├── main.py               # Script principal para execução do modelo
│   ├── model.py              # Definição do modelo LSTM Bidirecional
│   ├── data_preprocessing.py # Preparação e limpeza dos dados
│   ├── utilities.py          # Funções auxiliares
│   └── app.py                # Aplicativo Streamlit
│
├── notebooks/                # Notebooks Jupyter
│   ├── Exploratory_Data_Analysis.ipynb
│   └── Model_Training_and_Evaluation.ipynb
│
├── docs/                     # Documentação adicional
│   ├── setup.md              # Instruções de configuração
│   └── usage.md              # Instruções de uso
│
├── data/                     # Dados utilizados no projeto
│   ├── raw/                  # Dados brutos
│   ├── processed/            # Dados processados
│   ├── training/             # Dados de treinamento
│   ├── validation/           # Dados de validação
│   ├── test/                 # Dados de teste
│
└── models/                   # Modelos treinados e checkpoints 
```

## 📄 Published Articles

- [Machine Learning for Automatic Weather Stations: A Case Study](https://link.springer.com/chapter/10.1007/978-3-031-38344-1_6)
- [Package Proposal for Data Pre-Processing for Machine Learning Applied to Precision Irrigation](https://ieeexplore.ieee.org/abstract/document/10084899)
- [A Rapid Review on the Use of Free and Open Source Technologies and Software Applied to Precision Agriculture Practices](https://www.mdpi.com/2224-2708/12/2/28)
- [Analysis of MQTT-SN and LWM2M communication protocols for precision agriculture IoT devices](https://ieeexplore.ieee.org/abstract/document/9820048)
- [Título do Artigo 2]([Link para o artigo (DOI ou URL)]) 

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git

1. Create and activate a virtual environment:
  ```bash
python3 -m venv .venv
source .venv/bin/activate
 ```
2. Install the dependencies:
  ```bash
pip install -r requirements.txt
 ```

3. See the ```docs/setup.md  ``` file for detailed instructions on setting up the environment and running the code.

## 🤝 Collaboration
Feel free to get in touch if you have any questions, suggestions, or are interested in collaborating!
 ```bash
[ROGERIO PEREIRA DOS SANTOS]<br>
[rogerio.dosantos@ifpr.edu.br]<br>
```
