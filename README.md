# -pp5
Projektinis darbas nr 5
Gilus Neuroninis Tinklas su TF/Keras ir fast.ai
Apžvalga
Šiame projekte atliktas giluminio neuroninio tinklo treniravimas ir derinimas naudojant dvi skirtingas sistemas: TensorFlow/Keras ir fast.ai. Projektas sukurtas naudojant sintetinį duomenų rinkinį, kuris remiasi daugiamatės polinominės regresijos modeliu. Tikslas buvo palyginti šių dviejų sistemų modelių efektyvumą ir tikslumą, atsižvelgiant į įvairius našumo rodiklius.

Projekto Tikslai
Sintetinių Duomenų Generavimas: Sukurti duomenis, remiantis daugiamatės polinominės regresijos modeliu.
Modelio Treniravimas:
Pirmas modelis buvo sukurtas naudojant TensorFlow/Keras.
Antras modelis buvo sukurtas naudojant fast.ai biblioteką.
Modelių Palyginimas: Išanalizuoti ir palyginti modelių našumą pagal jų prognozuojamas reikšmes, tikslumą ir nuostolius.
Rezultatų Vizualizacija: Parodyti tikrąsias ir prognozuotas reikšmes abiem modeliams naudojant grafikus.
Kodo Struktūra
1. Sintetinių Duomenų Generavimas
Naudojama funkcija generate_polynomial_data(), kuri sukuria duomenis, remiasi polinominės regresijos modeliu:

python
Kopijuoti
def generate_polynomial_data(n_samples=1000, noise=0.1):
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=(n_samples, 2))  # Du įvesties bruožai
    y = (
        5 * X[:, 0] ** 3 + 2 * X[:, 1] ** 2 - 3 * X[:, 0] * X[:, 1] + noise * np.random.randn(n_samples)
    )
    return X, y
2. TensorFlow/Keras Modelis
TensorFlow/Keras modelis sukuriamas naudojant Sequential klasę, pridedant konvoliucinius ir pilnai sujungtus sluoksnius:

python
Kopijuoti
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model_tf = Sequential([
    Dense(64, activation="relu", input_dim=2),
    Dense(64, activation="relu"),
    Dense(1)  # Išvestis
])

model_tf.compile(optimizer=Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

history_tf = model_tf.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, verbose=0
)
3. fast.ai Modelis
fast.ai modelis sukuriamas naudojant TabularDataLoaders ir tabular_learner funkcijas:

python
Kopijuoti
from fastai.tabular.all import *

data = np.column_stack([X, y])
df = pd.DataFrame(data, columns=["feature1", "feature2", "target"])

dls = TabularDataLoaders.from_df(df, path=".", y_names="target", cont_names=["feature1", "feature2"],
                                 valid_idx=range(len(X_train), len(X)))

learn = tabular_learner(dls, layers=[64, 64], metrics=mae)
learn.fit_one_cycle(10, lr_max=1e-2)
4. Rezultatų Vizualizacija
Po modelio treniravimo, abiejų modelių prognozės buvo vizualizuotos palyginant tikras ir prognozuotas reikšmes:

python
Kopijuoti
plt.figure(figsize=(12, 6))

# TensorFlow/Keras rezultatai
plt.subplot(1, 2, 1)
plt.scatter(y_test, model_tf.predict(X_test).flatten(), alpha=0.5, label="Prognozuota")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", label="Idealus")
plt.title("TensorFlow/Keras Prognozės")
plt.xlabel("Tikros Vertės")
plt.ylabel("Prognozuotos Vertės")
plt.legend()

# fast.ai rezultatai
plt.subplot(1, 2, 2)
preds_np = fastai_preds.numpy().flatten()  # Pasirinkite visas prognozes ir suflatuokite
plt.scatter(y_test, preds_np[:len(y_test)], alpha=0.5, label="Prognozuota")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", label="Idealus")
plt.title("fast.ai Prognozės")
plt.xlabel("Tikros Vertės")
plt.ylabel("Prognozuotos Vertės")
plt.legend()

plt.tight_layout()
plt.show()
Modelio Palyginimas
Po atlikto palyginimo, pasirinktas TensorFlow/Keras modelis, nes jis prognozuoja reikšmes, kurios daug labiau atitinka tikras reikšmes. TensorFlow/Keras pasižymi didesniu lankstumu ir leidžia lengviau pritaikyti modelius sudėtingesniems uždaviniams. Be to, TensorFlow palaiko daugiau platformų ir aparatūros, tokių kaip GPU, TPU, mobiliosios platformos ir debesų kompiuterijos paslaugos.

Išvados
TensorFlow/Keras modelis pasirodė esąs tiksliau prognozuojantis ir suteikiantis daugiau galimybių modelio optimizavimui ir pritaikymui.
fast.ai taip pat gerai atliko užduotį, tačiau pasižymėjo mažesniu lankstumu ir optimizavimo galimybėmis.
Reikalingos Bibliotekos
Norėdami paleisti projektą, turite įdiegti šias bibliotekas:

bash
Kopijuoti
pip install tensorflow fastai scikit-learn matplotlib
Projekto Pradžia
Duomenų Paruošimas: Paleiskite sintetinį duomenų generavimo kodą, kad gautumėte polinominį duomenų rinkinį.
Modelio Treniravimas: Treniruokite modelius naudodami TensorFlow/Keras ir fast.ai sistemas.
Palyginimas ir Vizualizacija: Patikrinkite modelių prognozes ir įvertinkite jų tikslumą.

