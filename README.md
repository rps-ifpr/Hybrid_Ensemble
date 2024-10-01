# Hybrid_Ensemble
 Hybrid Ensemble Learning Approach for Real-Time Precision Irrigation in Hop Production
/projeto-estrutural-arquivos/
│
├── .git/                     # Controle de versão Git
├── .venv/                    # Ambiente virtual Python para isolamento de dependências
├── .idea/                    # Configurações do IDE (IntelliJ, PyCharm, etc.)
├── .gitignore                # Arquivos e pastas a serem ignorados pelo Git
├── README.md                 # Documentação inicial do projeto
├── requirements.txt          # Dependências do projeto
├── LICENSE                   # Licença do projeto
│
├── src/                      # Códigos-fonte do projeto
│   ├── main.py               # Script principal para executar o modelo
│   ├── model.py              # Definições do modelo LSTM Bidirecional
│   ├── data_preprocessing.py # Scripts para limpeza e preparação dos dados
│   ├── utilities.py          # Funções auxiliares
│   └── app.py                # Aplicativo Streamlit
│
├── notebooks/                # Notebooks Jupyter
│   ├── Exploratory_Data_Analysis.ipynb
│   └── Model_Training_and_Evaluation.ipynb
│
├── docs/                     # Documentação adicional
│   ├── setup.md
│   └── usage.md
│
├── data/                     # Dados do projeto
│   ├── raw/                  # Dados brutos
│   ├── processed/            # Dados processados
│   ├── training/             # Dados de treinamento
│   ├── validation/           # Dados de validação
│   ├── test/                 # Dados de teste
│
└── models/                   # Modelos treinados e checkpoints
