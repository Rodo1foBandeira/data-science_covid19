# Seasonal AutoRegressive Integrated Moving Average with eXogenous  (SARIMAX)
# Média móvel integrada autorregressiva sazonal com regressores exógenos
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

df = pd.read_csv('2020-04-26.csv', sep=';')
df_group = df.groupby('data')[['casosNovos', 'obitosNovos']].agg('sum')
df_group.index = pd.to_datetime(df_group.index).date
df_group = df_group.sort_index()

df_group.plot(title='Por dia')
plt.show()

df_group['casosAcumulado'] = np.cumsum(df_group['casosNovos'])
df_group['obitosAcumulado'] = np.cumsum(df_group['obitosNovos'])

df_group[['casosAcumulado', 'obitosAcumulado']].plot(title='Acumulado')
plt.show()


model = SARIMAX(df_group['obitosNovos'].values, exog= df_group['casosNovos'].values, trend='t', order=(1,1,0))
fit = model.fit()
predicts = fit.predict(0, len(df_group) + 8, exog= df_group['casosNovos'][-9:])

df_predicts = pd.DataFrame(data={ 'previsaoObitosDia': predicts }, index= [df_group.index[0] + datetime.timedelta(days= int(d)) for d in np.arange(len(predicts))])
df_predicts['realObitosDia'] = df_group['obitosNovos']

df_predicts.plot(title='Por dia')
plt.show()

df_predicts['realObitosAcumulado'] = np.cumsum(df_group['obitosNovos'])
df_predicts['previsaoObitosAcumulado'] = np.cumsum(df_predicts['previsaoObitosDia'])

df_predicts[['realObitosAcumulado', 'previsaoObitosAcumulado']].plot(title='Acumulado')
plt.show()
