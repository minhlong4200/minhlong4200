# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
​
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
​
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
​
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
​
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
add Codeadd Markdown
import numpy as np 
import pandas as pd
​
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_curve, precision_recall_curve, auc
from plotly.subplots import make_subplots
import itertools
add Codeadd Markdown
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
add Codeadd Markdown
missing_values_count = df.isnull().sum()
missing_values_count
add Codeadd Markdown
df.drop(['id','Unnamed: 32'],axis=1,inplace=True)
add Codeadd Markdown
fig = go.Figure(data=[go.Pie(labels=['Benign','Malignant'], values=df['diagnosis'].value_counts(), textinfo='label+percent')])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=['gold', 'mediumturquoise'], line=dict(color='#000000', width=2)))
fig.show()
add Codeadd Markdown
TRAIN_DIR = '../input/breast-cancer-wisconsin-data/data.csv'
add Codeadd Markdown
train = pd.read_csv(TRAIN_DIR, sep=',', header=0)
train = train.drop(['id', 'Unnamed: 32'], axis = 1)
add Codeadd Markdown
train.head()
​
add Codeadd Markdown
train.shape
add Codeadd Markdown
train.describe()
​
add Codeadd Markdown
X = train.drop(['diagnosis'], axis=1)
y = train['diagnosis'].apply(lambda x: 1 if x=='M' else -1)
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
add Codeadd Markdown
import matplotlib.pyplot as plt
import seaborn as sns
add Codeadd Markdown
sns.countplot(y)
​
add Codeadd Markdown
fig = plt.figure(figsize=(24, 18))
for i in range(len(X.columns)):
    plt.subplot(5, 6, i+1)
    plt.title(X.columns[i])
    plt.hist(X[X.columns[i]][y==-1], bins=25, color='lightblue', label='B-healthy')
    plt.hist(X[X.columns[i]][y==1], bins=25, color='grey', label='M-bad')
add Codeadd Markdown
from sklearn.model_selection import cross_val_score
​
add Codeadd Markdown
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6)
cross_val_score(dt, X, y, cv=8).mean()
add Codeadd Markdown
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20)
cross_val_score(rf, X, y, cv=8).mean()
add Codeadd Markdown
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=200)
cross_val_score(ada, X, y, cv=8).mean()