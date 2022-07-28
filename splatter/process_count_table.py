import pandas as pd
import numpy as np
import random

#read csv data

df = pd.read_csv("counts_splatter_de.prob=0.3.csv")
df = df.transpose()

names = pd.read_csv("types_splatter_de.prob=0.3.csv")
types_correct = list(names["Group"])

df.columns = list(df.iloc[0])
df = df.iloc[1:]

print(type(types_correct))

fraction_group1_to_g1g3 = 0.8
fraction_group2_to_g2g4 = 0.5

#replace some occurences of group1 by group3 according to fraction_group1_to_g1g3
#Likewise with group 2 and 4
#Stupid way to do so, but easier than fiddling with indices and random generators
"""
for i, g in enumerate(types_correct):
    #only have group 1 and 2 present
    if g == "Group1" and random.uniform(0, 1) > fraction_group1_to_g1g3:
        types_correct[i] = "Group3"
    elif g == "Group2" and random.uniform(0, 1) > fraction_group2_to_g2g4:
        types_correct[i] = "Group4"


"""

df.insert(0, "cell_type", types_correct)

#create dummy columns; don't need to alter pre-processing steps from my other script
df.insert(0, "Unnamed: 0", "") 
df.insert(0, "sample", "") 

#df.insert(0, "dummy", np.arange(len(df)))
df.set_index("sample", inplace = True)
"""
Missing: find highly variable genes
Would be good practice; however, we only simulated 50 genes from the beginning. That are few enough to handle
"""

df.to_csv("../flat_metric/data/genes_raw_splatter_untersch_Anzahl.csv")
