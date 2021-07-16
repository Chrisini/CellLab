import pandas as pd

df = pd.read_csv('squad.csv') # would need a csv file
print("CSV table:")
print(df)
print("Filter 'Kathi':")
filter = df[df.Name == "Kathi"]
print(filter)

df.ix[2, 2]