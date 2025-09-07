import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("network_traffic.csv")

# sns.boxenplot(data=df)
# plt.show()

print(df.info())
x=df[["Duration","SourcePort","DestinationPort","PacketCount","ByteCount"]]

le=LabelEncoder()
y=le.fit_transform(df["Label"])

f=df[["Protocol"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")

en=ohe.fit_transform(f)
get=ohe.get_feature_names_out(f.columns)
encode_data=pd.DataFrame(en,columns=get)

feature=df[["SourceIP","DestinationIP"]].copy()
for col in feature.columns:
    feature[col] = le.fit_transform(feature[col])

X=pd.concat([x,feature],axis=1)
X_final=pd.concat([X,encode_data],axis=1)
ss=StandardScaler()

X_s=ss.fit_transform(X_final)
x_train,x_test,y_train,y_test=train_test_split(X_s,y,test_size=0.2,random_state=42)

knc=KNeighborsClassifier(n_neighbors=10)
knc.fit(x_train,y_train)

print(knc.score(x_test,y_test))

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # relevant only if metric='minkowski'
}

gd=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=param_grid)
gd.fit(x_train,y_train)

print(gd.best_score_)