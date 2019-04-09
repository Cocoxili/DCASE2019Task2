import numpy as np
import pandas as pd

wavelist2018 = pd.read_csv("wavelist2018.csv")['fname'].tolist()
wavelist2019 = pd.read_csv("wavelist.csv")['fname'].tolist()

# print(wavelist2018)
# print(wavelist2019)

for fn in wavelist2019:
    if fn in wavelist2018:
        print(fn)


# print(result)
