# Detecção de Anomalias em Transações Financeiras

Este projeto tem como objetivo realizar a detecção de anomalias (fraudes) em um conjunto de dados de transações com cartão de crédito, utilizando técnicas de pré-processamento de dados e aprendizado de máquina.

## Pré-requisitos
Para executar este projeto, você precisará ter o Python instalado e as seguintes bibliotecas:

```bash
pip install pandas numpy scikit-learn
````

## Pode-se utilizar Notebook
1. Abra o Google Colab (ou Jupyter Notebook).
2. Cole o código em uma nova célula.

## Código
### Importação e Preparação
#### Carregamos os dados e realizamos a exploração inicial.
````

import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Lê o arquivo CSV da internet e o transforma em um DataFrame (tabela).
df = pd.read_csv(url) 

# Exibe as 5 primeiras linhas do DataFrame para verificar se os dados carregaram corretamente.
df.head() 
````

### Lidando com o desbalanceamento
````
# O parâmetro 'normalize=True' mostra a porcentagem em vez da contagem bruta.
# Isso é crucial porque em fraudes, 99.9% dos dados são transações normais, 
# então precisamos saber quão raro é o evento que queremos detectar.

df["Class"].value_counts(normalize=True)
````

### Engenharia de Recursos (Feature Engineering)
#### Aqui o objetivo é "preparar" os dados para que o modelo consiga entender melhor as diferenças entre valores.
````
import numpy as np # Importa o NumPy para cálculos matemáticos.

# A função np.log1p (log(1+x)) ajuda a reduzir a influência de valores muito grandes. 
# Transações financeiras costumam ter valores muito díspares, o log normaliza essa escala.

df["Amount_log"] = np.log1p(df["Amount"])
````

````
# Importa o normalizador de escala.
from sklearn.preprocessing import StandardScaler

# O scaler padroniza os dados para que tenham média 0 e desvio padrão 1.
scaler = StandardScaler()

# Isso evita que o modelo "se confunda" se uma coluna tiver números gigantes (como Amount) e outra coluna tiver números entre 0 e 1.
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
````

### Divisão dos Dados
````
from sklearn.model_selection import train_test_split 

# Cria a variável 'x' contendo tudo, menos a resposta final.
x = df.drop("Class", axis=1) 

# Cria a variável 'y' apenas com a coluna que queremos prever (Fraude ou Não).
y = df["Class"] 

# Divide em treino (70%) e teste (30%).
# 'stratify=y' garante que a proporção de fraudes seja a mesma nos dois grupos.
# 'random_state=42' garante que, se você rodar de novo, o resultado seja o mesmo.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, stratify=y, test_size=0.3, random_state=42
)
````
