import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

with open('history.json', 'r') as f:
    history = json.load(f)

df = pd.DataFrame(
    data=history
)

sns.violinplot(df)
plt.show()
