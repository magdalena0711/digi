import pandas as pd
import numpy as np
# import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import pandas as pd

df1 = pd.read_csv('nulls_deleted.csv', parse_dates=['measure_date'])
df2 = pd.read_csv('NIS_ESP_events_anonymized.csv', parse_dates=['date_time'])

# Make sure the column names match
df1.rename(columns={'measure_date': 'date_time'}, inplace=True)

# Merge on 'well'
merged = pd.merge(df1, df2, on='well', suffixes=('_telemetry', '_events'))

# Filter for the first event after the telemetry date
result = merged[merged['date_time_events'] > merged['date_time_telemetry']]

# Now group by 'well' and find the first event for each telemetry date
result = result.sort_values(['well', 'date_time_telemetry', 'date_time_events'])
result = result.groupby(['well', 'date_time_telemetry']).first().reset_index()

# Rename the columns for clarity
result.rename(columns={'date_time_events': 'first_event_date_time'}, inplace=True)
df = result

# Učitavanje podataka iz CSV fajla
# df = pd.read_csv('nulls_deleted.csv')  # Zamenite sa stvarnim imenom fajla

# Definisanje karakteristika (X) i ciljne promenljive (y)
#df['days'] = df['days'].str.replace(' days', '').astype(int)

df['real'] = df['freq_radno_opterecenje'].apply(lambda z: z.real)  # Extract real part
df['imag'] = df['freq_radno_opterecenje'].apply(lambda z: z.imag) 

X = df[['napon_ab', 'napon_bc', 'napon_ca',
       'elektricna_struja_fazaa', 'elektricna_struja_fazab',
       'elektricna_struja_fazac', 'koeficijent_kapaciteta',
       'frekvencija', 'radno_opterecenje',
       'aktivna_snaga', 'pritisak_na_prijemu_pumpe',
       'temperatura_motora', 'temperatura_u_busotini', 'real', 'imag']]
y = df['label']  # Kolona koja sadrži tip kvara

# Deljenje na trening i test setove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pipeline za obradu numeričkih vrednosti
numeric_features = X.columns
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Kreiranje preprocesora
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)], remainder='passthrough')

# Kreiranje klasifikatora (Random Forest)
model = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(n_estimators=100, 
                                                               random_state=42,
                                                               class_weight='balanced'))])


# Treniranje modela
model.fit(X_train, y_train)

# Predikcija na test podacima
y_pred = model.predict(X_test)

# Kreiraj SHAP eksplainer
#model_instance = model.named_steps['classifier']  # Izvuci model
#explainer = shap.TreeExplainer(model_instance)

# Izračunaj SHAP vrednosti za trening podatke

# Izračunaj SHAP vrednosti za trening podatke
#X_train_transformed = preprocessor.transform(X_train)  # Transformišite podatke
#hap_values = explainer.shap_values(X_train_transformed)

# Prikazivanje naziva karakteristika
#feature_names = np.concatenate([numeric_features])  # Dodajte nazive karakteristika
#shap.summary_plot(shap_values, X_train_transformed, feature_names=feature_names, max_display=len(feature_names))


# Prikazi pojedinačne SHAP vrednosti za jedan uzorak
#shap.initjs()  # Inicijalizuj JavaScript za vizualizaciju
#shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train_transformed[0], feature_names=feature_names)

# Evaluacija modela
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Prikazivanje konfuzione matrice
# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize  # New import for binarizing the labels

# Binarize the output (for multi-class ROC AUC)
y_test_binarized = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_binarized.shape[1]

# Get the predicted probabilities
y_prob = model.predict_proba(X_test)

# Calculate the ROC AUC for each class and average them
roc_auc = {}
for i in range(n_classes):
    roc_auc[i] = roc_auc_score(y_test_binarized[:, i], y_prob[:, i])

# Average the AUC scores
roc_auc_average = np.mean(list(roc_auc.values()))
print(f"ROC AUC (average): {roc_auc_average:.4f}")

# Generating ROC curve
plt.figure()
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multi-class')
plt.legend(loc='lower right')
plt.grid()
plt.show()