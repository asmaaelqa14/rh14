import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Définition des colonnes
COLONNES_CATEGORIELLES = ["EducationField", "BusinessTravel", "EnvironmentSatisfaction", "MaritalStatus"]
COLONNES_CONTINUES = ["Age", "TotalWorkingYears", "YearsAtCompany", "YearsWithCurrManager", "meanPresenceTime"]
AUTRES_COLONNES = ['Education', 'JobLevel', 'JobInvolvement', 'Department', 'JobSatisfaction', 'JobRole', 'NumCompaniesWorked', 'WorkLifeBalance']
TOUTES_LES_COLONNES = COLONNES_CATEGORIELLES + COLONNES_CONTINUES + AUTRES_COLONNES

# Codes groupés
efg_code = {
    "Life Sciences": "Other", "Human Resources": "Human Resources ",
    "Marketing": "Other", "Technical Degree": "Technical Degree",
    "Other": "Other", "Medical": "Other"
}
jrg_code = {
    "Healthcare Representative": "Other", "Laboratory Technician": "Other",
    "Human Resources": "Other", "Manager": "Other", "Manufacturing Director": "Manufacturing Director",
    "Research Director": "Research Director", "Research Scientist": "Other",
    "Sales Executive": "Other", "Sales Representative": "Other"
}

# Classe pour l'ingénierie des features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def encode(series, code):
            return pd.Series(series).map(code).values

        return np.c_[
            (X['Education'] == 2).values,
            (X['JobLevel'] == 2).values,
            (X['JobInvolvement'] == 1).values,
            (X['EnvironmentSatisfaction'] == 1).values,
            (X['WorkLifeBalance'] == 1).values,
            (X['Department'] == "Human Resources").values,
            pd.Series(X["JobSatisfaction"]).map({1: 1, 2: 2, 3: 2, 4: 3}).values,
            encode(X["EducationField"], efg_code),
            encode(X["JobRole"], jrg_code),
            ((X['MaritalStatus'] == 'Single') & (X["BusinessTravel"] != "Non-Travel")).values,
            ((X['MaritalStatus'] != 'Single') & (X["BusinessTravel"] != "Non-Travel")).values,
            (X['meanPresenceTime'] > 8).values,
            ((X['Age'] <= 35) & (X['NumCompaniesWorked'] > 3)).values,
            ((X['Age'] <= 35) & (X["BusinessTravel"] != "Non-Travel")).values,
        ]

# Fonction utilitaire pour charger les objets joblib
def charger_joblib(fichier, nom_affichage):
    try:
        return joblib.load(fichier)
    except FileNotFoundError:
        st.error(f"Le fichier {fichier} est introuvable. Veuillez le télécharger sur GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement de {nom_affichage} : {e}")
        st.stop()

# Prétraitement des données utilisateur
def preprocess_user_input(data):
    df = pd.DataFrame([data], columns=TOUTES_LES_COLONNES)
    
    # Charger une seule fois les objets
    imputer = charger_joblib('imputer.joblib', 'imputer')
    scaler = charger_joblib('scaler.joblib', 'scaler')
    encoder_disc = charger_joblib('encoder_disc.joblib', 'encoder_disc')
    
    # Traitement des variables continues
    cont_processed = imputer.transform(df[COLONNES_CONTINUES])
    cont_scaled = scaler.transform(cont_processed)
    cont_scaled_df = pd.DataFrame(cont_scaled, columns=COLONNES_CONTINUES)
    
    # Traitement des variables catégorielles
    disc_encoded = encoder_disc.transform(df[COLONNES_CATEGORIELLES])
    disc_processed_df = pd.DataFrame(disc_encoded, 
                                     columns=encoder_disc.get_feature_names_out(COLONNES_CATEGORIELLES))
    
    # Ingénierie des features additionnelles
    feat_eng_data = CombinedAttributesAdder().transform(df)
    feat_eng_processed_df = pd.DataFrame(feat_eng_data, columns=[
        'EducationIs2', 'JobLevelIs2', 'JobInvolvementIs1', 'EnvironmentSatisfactionIs1',
        'WorkLifeBalanceIs1', 'DepartmentIsHumanResources', 'JobSatisfactionGrouped',
        'EducationFieldGrouped', 'JobRoleGrouped', 'SingleAndTravelling',
        'MarriedOnceAndTravelling', 'DoOvertime', 'IsYoungAndWorkedInManyCompanies',
        'IsYoungAndTravel'
    ])
    
    # Combinaison finale des features
    processed_data = pd.concat([disc_processed_df, cont_scaled_df, feat_eng_processed_df], axis=1)
    
    return processed_data

# Chargement du modèle et des données de référence
model = charger_joblib('modele_attrition.joblib', 'modèle')
hr_df_loaded = charger_joblib('hr_df_for_streamlit_values.joblib', 'données de référence')

# Interface utilisateur Streamlit
st.title("Prédiction de l'attrition des employés")
st.write("Veuillez entrer les informations de l'employé pour prédire le risque d'attrition.")

input_data = {}

# Options de sélection pour les variables catégorielles
select_options = {
    "EducationField": ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'],
    "BusinessTravel": ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],
    "EnvironmentSatisfaction": [3.0, 2.0, 4.0, 1.0, float('nan')],
    "MaritalStatus": ['Married', 'Single', 'Divorced'],
    "Department": ['Sales', 'Research & Development', 'Human Resources'],
    "JobRole": ['Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Human Resources',
                'Research Director', 'Laboratory Technician', 'Manufacturing Director', 'Sales Representative', 'Manager']
}

for col, options in select_options.items():
    input_data[col] = st.selectbox(f"Sélectionnez {col}", options)

# Saisie des variables continues
for col in COLONNES_CONTINUES:
    input_data[col] = st.number_input(
        f"Entrez {col}",
        min_value=float(hr_df_loaded[col].min()),
        max_value=float(hr_df_loaded[col].max())
    )

# Sliders pour d'autres variables numériques discrètes
slider_cols = {
    'Education': (1, 5),
    'JobLevel': (1, 5),
    'JobInvolvement': (1, 4),
    'JobSatisfaction': (1, 4),
    'WorkLifeBalance': (1, 4),
    'NumCompaniesWorked': (0, int(hr_df_loaded['NumCompaniesWorked'].max()))
}

for col, (min_v, max_v) in slider_cols.items():
    default = int(hr_df_loaded[col].mode()[0]) if col in hr_df_loaded else min_v
    input_data[col] = st.slider(f"Sélectionnez {col}", min_value=min_v, max_value=max_v, value=default)

# Bouton de prédiction
if st.button("Prédire l'attrition"):
    try:
        processed_input = preprocess_user_input(input_data)
        # Prédiction de la probabilité d'attrition (classe 'Yes')
        prediction_proba = model.predict_proba(processed_input)[0][1]
        prediction_threshold = 0.5
        prediction = "Oui" if prediction_proba >= prediction_threshold else "Non"

        st.subheader("Résultat de la prédiction")
        st.write(f"Probabilité d'attrition : {prediction_proba:.2f}")
        st.write(f"Prédiction d'attrition (seuil > {prediction_threshold}) : **{prediction}**")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
