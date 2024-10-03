# Hybrid_Ensemble
 Hybrid Ensemble Learning Approach for Real-Time Precision Irrigation in Hop Production

## Estrutura do Projeto

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