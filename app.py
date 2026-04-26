import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Cargar datos
df = pd.read_csv("./data/titanic_feature_engineering.csv")

# Variables predictoras y objetivo
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Crear pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Entrenar modelo
pipeline.fit(X_train, y_train)

# Ejemplo de pasajero
nuevo_pasajero = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 1,
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": 2
}])

# Predicción
prediccion = pipeline.predict(nuevo_pasajero)

# Resultado
if prediccion[0] == 1:
    print("El pasajero probablemente sobreviviría.")
else:
    print("El pasajero probablemente no sobreviviría.")