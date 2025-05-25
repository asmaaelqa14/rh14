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

# Classe pour ingénierie de features
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        def encode(series, code): return pd.Series(series).map(code).values
        return np.c_[
            (X['Education'] == 2).values,
            (X['JobLevel'] == 2).values,
            (X['JobInvolvement'] == 1).values,
            (X['EnvironmentSatisfaction'] == 1).values,
            (X['WorkLifeBalance'] == 1).values,
            (X['Department'] == "Human Resources").values,
            encode(X["JobSatisfaction"], {1: 1, 2: 2, 3: 2, 4: 3}),
            encode(X["EducationField"], efg_code),
            encode(X["JobRole"], jrg_code),
            ((X['MaritalStatus'] == 'Single') & (X["BusinessTravel"] != "Non-Travel")).values,
            ((X['MaritalStatus'] != 'Single') & (X["BusinessTravel"] != "Non-Travel")).values,
            (X['meanPresenceTime'] > 8).values,
            ((X['Age'] <= 35) & (X['NumCompaniesWorked'] > 3)).values,
            ((X['Age'] <= 35) & (X["BusinessTravel"] != "Non-Travel")).values,
        ]

# Fonction utilitaire pour chargement sécurisé
def charger_joblib(fichier, nom_affichage):
    try:
        return joblib.load(fichier)
    except FileNotFoundError:
        st.error(f"Le fichier {fichier} est introuvable. Veuillez le télécharger.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement de {nom_affichage} : {e}")
        st.stop()

# Prétraitement utilisateur
def preprocess_user_input(data):
    df = pd.DataFrame([data], columns=TOUTES_LES_COLONNES)
    cont_processed = charger_joblib('imputer.joblib', 'imputer').transform(df[COLONNES_CONTINUES])
    cont_scaled = charger_joblib('scaler.joblib', 'scaler').transform(cont_processed)
    disc_encoded = charger_joblib('encoder_disc.joblib', 'encoder_disc').transform(df[COLONNES_CATEGORIELLES])
    
    feat_eng_data = CombinedAttributesAdder().transform(df)
    
    df_final = pd.concat([
        pd.DataFrame(disc_encoded, columns=charger_joblib('encoder_disc.joblib', 'encoder_disc').get_feature_names_out(COLONNES_CATEGORIELLES)),
        pd.DataFrame(cont_scaled, columns=COLONNES_CONTINUES),
        pd.DataFrame(feat_eng_data, columns=[
            'EducationIs2', 'JobLevelIs2', 'JobInvolvementIs1', 'EnvironmentSatisfactionIs1',
            'WorkLifeBalanceIs1', 'DepartmentIsHumanResources', 'JobSatisfactionGrouped',
            'EducationFieldGrouped', 'JobRoleGrouped', 'SingleAndTravelling',
            'MarriedOnceAndTravelling', 'DoOvertime', 'IsYoungAndWorkedInManyCompanies',
            'IsYoungAndTravel'
        ])
    ], axis=1)
    
    return df_final

# Chargement des ressources
model = charger_joblib('modele_attrition.joblib', 'modèle')
hr_df_loaded = charger_joblib('hr_df_for_streamlit_values.joblib', 'données de référence')

# Interface utilisateur
st.title("Prédiction de l'attrition des employés")
st.write("Veuillez entrer les informations de l'employé pour prédire le risque d'attrition.")

input_data = {}

# Sélecteurs pour variables catégorielles avec valeurs pré-définies ou extraites
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

# Champs numériques continus
for col in COLONNES_CONTINUES:
    input_data[col] = st.number_input(
        f"Entrez {col}",
        min_value=float(hr_df_loaded[col].min()),
        max_value=float(hr_df_loaded[col].max())
    )

# Sliders pour les autres colonnes numériques discrètes
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

# Prédiction
if st.button("Prédire l'attrition"):
    input_df = preprocess_user_input(input_data)
    prediction = model.predict_proba(input_df)[0][1]  # probabilité d'attrition
    st.success(f"Probabilité d'attrition : {prediction:.2%}")
