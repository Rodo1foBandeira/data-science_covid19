# Média móvel integrada autorregressiva sazonal com regressores exógenos (SARIMAX)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ministerio_Saude.csv', sep=';')

df_obitos_dia = df.groupby('data')['obitosNovos'].agg('sum')

df_obitos_dia.index = pd.to_datetime(df_obitos_dia.index, format="%d/%m/%Y").date

df_obitos_dia = df_obitos_dia.sort_index()

df_infeccao_dia = df.groupby('data')['casosNovos'].agg('sum')

df_infeccao_dia.index = pd.to_datetime(df_infeccao_dia.index, format="%d/%m/%Y").date

df_infeccao_dia = df_infeccao_dia.sort_index()


model = SARIMAX(df_obitos_dia.values, exog=df_infeccao_dia.values, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)

yhat = model_fit.predict(len(df_obitos_dia), len(df_infeccao_dia) + 7, exog=df_infeccao_dia.values[-8:])