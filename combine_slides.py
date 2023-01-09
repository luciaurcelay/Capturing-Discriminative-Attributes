# %%
from pathlib import Path
import pandas as pd

# %%

path = Path("resources/conceptnet_relations")
# %%
path
# %%
list(path.glob("*"))
# %%
dfs = []
for file in list(path.glob("*")):
    df = pd.read_csv(file)
    print(df)
    dfs.append(df)
# %%
print(dfs)
# %%
pd.concat(dfs).to_csv("conceptnet.csv", index=False)
# %%
