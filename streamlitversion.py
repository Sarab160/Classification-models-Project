import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier

# --------------------------
# Load dataset
# --------------------------
df = pd.read_csv("network_traffic.csv")

# Encode label
le = LabelEncoder()
y = le.fit_transform(df["Label"])

# Base numeric features
x = df[["Duration", "SourcePort", "DestinationPort", "PacketCount", "ByteCount"]]

# Encode categorical Protocol
f = df[["Protocol"]]
ohe = OneHotEncoder(sparse_output=False, drop="first")
en = ohe.fit_transform(f)
encode_data = pd.DataFrame(en, columns=ohe.get_feature_names_out(f.columns))

# Encode IPs
feature = df[["SourceIP", "DestinationIP"]].copy()
for col in feature.columns:
    feature[col] = le.fit_transform(feature[col])

# Final features
X = pd.concat([x, feature], axis=1)
X_final = pd.concat([X, encode_data], axis=1)

# Scaling
ss = StandardScaler()
X_s = ss.fit_transform(X_final)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

# --------------------------
# Train models
# --------------------------
models = {
    "KNN": KNeighborsClassifier(n_neighbors=30),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "GaussianNB": GaussianNB(),
    "SVC (poly)": SVC(kernel="poly", probability=True),
    "Voting Classifier": VotingClassifier(estimators=[
        ('dtc', DecisionTreeClassifier()),
        ('gnb', GaussianNB()),
        ('knn', KNeighborsClassifier()),
        ('svc', SVC(probability=True))
    ]),
    "Bagging (SVC)": BaggingClassifier(estimator=SVC(), n_estimators=20),
    "Random Forest": RandomForestClassifier(n_estimators=20)
}

trained_models = {}
accuracies = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    trained_models[name] = model
    accuracies[name] = acc

# --------------------------
# Streamlit UI
# --------------------------
st.title("🚦 Network Traffic Classification Dashboard")

st.subheader("📊 Model Accuracies")
for model, acc in accuracies.items():
    st.write(f"**{model}:** {acc:.4f}")

st.subheader("🔮 Make a Prediction")

# User input form
with st.form("prediction_form"):
    duration = st.number_input("Duration", min_value=0.0, value=10.0)
    source_port = st.number_input("Source Port", min_value=0, value=80)
    destination_port = st.number_input("Destination Port", min_value=0, value=443)
    packet_count = st.number_input("Packet Count", min_value=0, value=100)
    byte_count = st.number_input("Byte Count", min_value=0, value=1024)
    source_ip = st.number_input("Source IP (encoded)", min_value=0, value=10)
    dest_ip = st.number_input("Destination IP (encoded)", min_value=0, value=20)
    protocol = st.selectbox("Protocol", ohe.categories_[0])

    submitted = st.form_submit_button("Submit Input")

if submitted:
    # Encode protocol
    proto_encoded = ohe.transform([[protocol]])
    proto_df = pd.DataFrame(proto_encoded, columns=ohe.get_feature_names_out(["Protocol"]))

    # Create single input row
    user_input = pd.DataFrame([[
        duration, source_port, destination_port, packet_count, byte_count, source_ip, dest_ip
    ]], columns=["Duration", "SourcePort", "DestinationPort", "PacketCount", "ByteCount", "SourceIP", "DestinationIP"])

    final_input = pd.concat([user_input, proto_df], axis=1)

    # Scale input
    final_input_scaled = ss.transform(final_input)

    st.subheader("⚡ Predictions")
    for name, model in trained_models.items():
        pred = model.predict(final_input_scaled)[0]
        label = le.inverse_transform([pred])[0]
        st.write(f"**{name} Prediction:** {label}")
