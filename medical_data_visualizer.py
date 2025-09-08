import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv("medical_examination.csv")

# 2  Cria a coluna overweight, usando np.where como condicional para calcular o IMC com peso/altura^2, converte e altura de cm para metros (dividindo por 100) 
df['overweight'] = np.where((df['weight'] / ((df['height'] / 100) ** 2)) > 25, 1, 0)

# 3 Para padronizar valores, 0 é bom (peso normal) e 1 é ruim (peso acima do normal)
def cholesterol(x):
    if x == 1:
        return 0
    else:
        return 1

def gluc(y):
    if y == 1:
        return 0
    else:
        return 1

# Ao usar apply, aplicamos a função em cada valor da coluna
df['cholesterol'] = df['cholesterol'].apply(cholesterol)
df['gluc'] = df['gluc'].apply(gluc)

# 4
def draw_cat_plot():
    # 5 com pd.melt, convertemos colunas em linhas. A coluna cardio fica fixa ao usar id_vars 
    df_cat = pd.melt(df, id_vars = ['cardio'], value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 Agrupa as variáveis, conta quantos registros existem em cada grupo e transforma o resultado em um df na coluna total
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')    

    # 7 e 8 Para criar gráficos categóricos, em que X possui as variáveis categóricas e Y a contagem de casos, hue separa por valor 0 ou 1, col cria painéis separados para pessoas com e sem problemas cardíacos, e .fig acessa a figura do matplot para salvar depois
    fig = sns.catplot(data=df_cat, x="variable", y="total", hue="value", kind="bar", col = "cardio").fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11 
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])& # Para remover os registros de pressão arterial invertida
                # Para remover outliers de altura e peso usando os percentis de 2,5 e 97,5
                 (df['height'] >= df['height'].quantile(0.025))&
                 (df['height'] <= df['height'].quantile(0.975))&
                 (df['weight'] >= df['weight'].quantile(0.025))&
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12 Para calcular a correlação entre todas as variáveis numéricas, em que valores próximos de 1 ou -1 tem forte correlação positiva ou negativa
    corr = df_heat.corr()

    # 13 Para esconder metade da parte superior do mapa de calor e evitar duplicatas, criamos uma máscara
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(8, 6))

    # 15 Para criar o mapa de calor com seaborn
    sns.heatmap(corr, annot=True, fmt=".1f", mask=mask, cmap='coolwarm', vmax=.25, vmin=-0.1, center=0.08, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # 16
    fig.savefig('heatmap.png')
    return fig
