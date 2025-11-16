# Sistema de Predição - Game Changers 2025: Championship Seoul

## 📋 Sobre o Projeto

Sistema de Machine Learning para predição das colocações dos times no campeonato Game Changers 2025 de Valorant. Desenvolvido seguindo princípios de TDD, SOLID e arquitetura limpa.

**Este projeto foi desenvolvido exclusivamente para fins educacionais e como um portfólio de engenharia de software e machine learning.**

## 🏗️ Arquitetura do Sistema

### Estrutura Modular
- **Data Layer**: Coleta e processamento de dados da API VLR.gg
- **Feature Layer**: Engenharia de features e pré-processamento
- **Model Layer**: Treinamento e predição de modelos de ML
- **Domain Layer**: Entidades e regras de negócio

### Princípios Aplicados
- **SRP**: Cada classe tem uma única responsabilidade
- **OCP**: Aberto para extensão, fechado para modificação
- **DIP**: Dependência de abstrações, não implementações
- **TDD**: Desenvolvimento guiado por testes

## 📁 Estrutura de Arquivos

### Diretório Principal
```bash
game_changers_ml/
├── src/ # Código fonte
│ ├── data/       # Coleta e processamento de dados
│ ├── features/   # Engenharia de features
│ ├── models/     # Modelos de ML
│ ├── domain/     # Entidades de domínio
│ ├── services/   # Orquestração e serviços
│ ├── utils/      # Utilitários e configuração
│ └── validation/ # Lógica de validação
├── tests/ # Testes unitários e de integração
│   ├── test_data.py           # Testes de dados
│   ├── test_features.py       # Testes de features
│   ├── test_models.py         # Testes de modelos
│   ├── test_integration.py    # Testes de integração
│   └── test_api.py            # Testes de API
├── config.yaml # Configurações do sistema
├── requirements.txt # Dependências do projeto
└── main.py # Script principal
```

### Descrição dos Arquivos

#### Configuração
- **config.yaml**: Configurações de API, features e modelo
- **requirements.txt**: Dependências do Python

#### Camada de Dados
- **src/data/collectors.py**: Coleta dados da API VLR.gg
- **src/data/processors.py**: Processa dados brutos em features

#### Camada de Features
- **src/features/engineering.py**: Engenharia e pré-processamento de features

#### Camada de Modelo
- **src/models/trainer.py**: Treinamento e tuning de modelos
- **src/models/predictor.py**: Predição de colocações

#### Domínio
- **src/domain/entities.py**: Entidades do domínio (Team, Player, Match)
- **src/domain/tournament.py**: Gerencia as configurações e o acesso aos dados dos torneios

#### Serviços
- **src/services/prediction_service.py**: Orquestra o fluxo de treinamento, predição e validação.

#### Validação
- **src/validation/validator.py**: Contém a lógica para validar a performance do modelo contra dados históricos.

#### Utilitários
- **src/utils/config.py**: Gerenciamento de configurações

#### Testes
- **tests/test_*.py**: Testes unitários para cada módulo
- **tests/test_integration.py**: Testes de integração

## 🚀 Como Executar

### 1. Instalação
```bash
pip install -r requirements.txt
```

### 2. Execução
```bash
python main.py
```

### 3. Testes
```bash
pytest tests/ -v
```

## 🔧 Funcionalidades

### Features Calculadas

- **Estabilidade do Elenco**: Tempo que o time está junto
- **Performance Individual**: Rating médio dos jogadores
- **Performance do Time**: Win rate e forma recente
- **Força da Agenda**: Qualidade dos oponentes enfrentados

### Modelos de ML

- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

## 📊 Métricas de Avaliação

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Accuracy (±1 posição)
- Cross-validation scores
- Confidence scores para predições

## 🔮 Próximos Passos

1. **Evoluir a Integração da API**: Migrar para uma fonte de dados oficial/de produção e implementar um sistema de cache robusto.
2. **Aprofundar Engenharia de Features**: Adicionar estatísticas mais granulares, como performance por mapa, dados de economia e estatísticas de agentes.
3. **Implementar Ensemble de Modelos**: Combinar as predições de múltiplos modelos (ex: Stacking, Voting) para aumentar a acurácia e a robustez do resultado final.
4. **Deployment em Nuvem**: Containerizar a aplicação (Docker) e implantá-la em um ambiente de nuvem (AWS, GCP, Azure) para execução automatizada.

## 👨‍💻 Desenvolvimento

Este projeto foi desenvolvido seguindo:

- Test-Driven Development (TDD)
- Princípios SOLID
- Arquitetura Limpa
- Versionamento semântico

## 🎯 Considerações Finais

### Princípios Aplicados

1. **TDD**: Todos os componentes possuem testes unitários

2. **SOLID**: 
   - SRP: Cada classe tem responsabilidade única
   - OCP: Fácil extensão para novos modelos/features
   - LSP: Interfaces consistentes
   - ISP: Interfaces segregadas e específicas
   - DIP: Injeção de dependências

3. **Arquitetura Limpa**: Separação clara entre camadas

### Escalabilidade
- Fácil adição de novos modelos
- Configuração centralizada
- Processamento modular de features
- Sistema de fallback para predições