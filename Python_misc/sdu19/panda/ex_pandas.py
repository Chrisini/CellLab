import pandas as pd

#df = dataframe
df = pd.read_csv("auto.csv")

#print(file)

filter = df[df.mpg > 15.9]
#print(filter)

a = df[:7]
print("a: row 0 to 6")
#print(a)
b = pd.DataFrame(a, columns = ["weight", "acceleration"])
print("b: weight and acceleration:")
print(b)
#or in one step (.ix will not be available in the future)
#c = df.ix[:7, ("weight", "acceleration")]
#print(c)

filter2 = df.loc[df["horsepower"] != '?']
df['horsepower'] = filter2.horsepower.astype(int)
d = pd.DataFrame(filter2, columns = ["mpg", "horsepower"])
print(d)


e = df.describe()
print("e: describe")
print(e)
