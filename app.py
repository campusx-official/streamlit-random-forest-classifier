import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

def draw_meshgrid():
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

df = pd.read_csv('concertriccir2.csv')
X = df.iloc[:,0:2].values
y = df.iloc[:,-1].values
#X, y = make_circles(n_samples=500, factor=0.1, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Random Forest Classifier")

n_estimators = int(st.sidebar.number_input('Num Estimators'))

max_features = st.sidebar.selectbox(
    'Max Features',
    ('auto', 'sqrt','log2','manual')
)

if max_features == 'manual':
    max_features = int(st.sidebar.number_input('Max Features'))

bootstrap = st.sidebar.selectbox(
    'Bootstrap',
    ('True', 'False')
)

max_samples = st.sidebar.slider('Max Samples', 1, X_train.shape[0], 1,key="1236")

# Load initial graph
fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
orig = st.pyplot(fig)

if st.sidebar.button('Run Algorithm'):

    orig.empty()

    if n_estimators == 0:
        n_estimators = 100

    clf = RandomForestClassifier(n_estimators=n_estimators,random_state=42,bootstrap=bootstrap,max_samples=max_samples,max_features=max_features)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid()
    labels = clf.predict(input_array)

    ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)
    st.header("Accuracy - " + str(round(accuracy_score(y_test, y_pred), 2)))
