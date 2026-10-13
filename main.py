
import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Lê o arquivo CSV da internet e o transforma em um DataFrame (tabela).
df = pd.read_csv(url) 

# Exibe as 5 primeiras linhas do DataFrame para verificar se os dados carregaram corretamente.
df.head() 
