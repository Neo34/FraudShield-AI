# FraudShield-# FraudShield AI

Sistema de detecção de fraudes financeiras utilizando Machine Learning e técnicas avançadas de balanceamento de dados.

## Objetivo

Detectar transações fraudulentas em grandes volumes de dados utilizando algoritmos supervisionados de Machine Learning.

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- SMOTE
- Matplotlib

## Arquitetura

DataLoader
↓
Preprocessor
↓
SMOTE
↓
Random Forest
↓
Evaluation
↓
SHAP Explainability

## Dataset

Credit Card Fraud Detection Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Funcionalidades

- Carregamento de dados
- Balanceamento de classes
- Treinamento de modelo
- Avaliação
- Explicabilidade com SHAP
- Predição de novas transações

## Como Executar

```bash
pip install -r requirements.txt
```

```bash
python src/main.py
```

## Resultados

Métricas avaliadas:

- Accuracy
- Precision
- Recall
- F1 Score

## Melhorias Futuras

- API FastAPI
- Dashboard Streamlit
- Deploy Azure ML
- Docker
- CI/CD GitHub Actions
