import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder,StandardScaler,PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier,BaggingClassifier,RandomForestClassifier

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

knc=KNeighborsClassifier(n_neighbors=30)
knc.fit(x_train,y_train)

print(knc.score(x_test,y_test))



pe=PolynomialFeatures(degree=1)
x_pe=pe.fit_transform(X_s)
x_train1,x_test1,y_train1,y_test1=train_test_split(x_pe,y,test_size=0.3,random_state=42)
lr=LogisticRegression()
lr.fit(x_train1,y_train1)

print(lr.score(x_test1,y_test1))

dtc=DecisionTreeClassifier(max_depth=5)
dtc.fit(x_train,y_train)

print(dtc.score(x_test,y_test))

gnb=GaussianNB()
gnb.fit(x_train,y_train)

print(gnb.score(x_test,y_test))

svc=SVC(kernel="poly")
svc.fit(x_train,y_train)

print(svc.score(x_test,y_test))

models = [
    ('dtc', DecisionTreeClassifier()),
    ('gnb', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('svc',SVC())]

voting=VotingClassifier(estimators=models)
voting.fit(x_train,y_train)

print(voting.score(x_test,y_test))

bgc=BaggingClassifier(estimator=SVC(),n_estimators=20)
bgc.fit(x_train,y_train)

print(bgc.score(x_test,y_test))

rf=RandomForestClassifier(n_estimators=20)
rf.fit(x_train,y_train)

print(rf.score(x_test,y_test))