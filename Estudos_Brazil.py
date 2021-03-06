import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cases_time.csv')

df_brazil = df.loc[df['Country_Region'] == 'Brazil']

df_brazil.index = pd.to_datetime(df_brazil['Report_Date_String'], format="%Y/%m/%d")

delta_death = [0]
i = 0
for value in df_brazil['Deaths']:
   if i > 0:
      delta_death.append(value - df_brazil['Deaths'][i-1])
   i+=1

df_brazil['Delta_Death'] = delta_death

df_brazil[['Confirmed', 'Deaths', 'Delta_Confirmed', 'Delta_Deaph']].plot()

plt.show()