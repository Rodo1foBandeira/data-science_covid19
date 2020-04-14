import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cases_time.csv')

df_brazil = df.loc[df['Country_Region'] == 'Brazil']

df_brazil.index = pd.to_datetime(df_brazil['Report_Date_String'], format="%Y/%m/%d")

df_brazil[['Confirmed', 'Deaths', 'Delta_Confirmed']].plot()

plt.show()

df_brazil['Delta_Confirmed'].plot(legend=True)

plt.show()
