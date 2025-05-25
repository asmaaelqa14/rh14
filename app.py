import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Liste des colonnes nécessaires pour le modèle (à adapter précisément)
COLONNES_CATEGORIELLES = ["EducationField", "BusinessTravel", "EnvironmentSatisfaction", "MaritalStatus"]
COLONNES_CONTINUES = ["Age", "TotalWorkingYears", "YearsAtCompany", "YearsWithCurrManager", "meanPresenceTime"]
TOUTES_LES_COLONNES = COLONNES_CATEGORIELLES + COLONNES_CONTINUES + ['Education', 'JobLevel', 'JobInvolvement', 'Department', 'JobSatisfaction', 'JobRole', 'NumCompaniesWorked'] # Ajoute toutes les colonnes utilisées dans ton pipeline de features

# Codes pour les features groupées (tirés de ton code)
efg_code = {
    "Life Sciences": "Other",
    "Human Resources": "Human Resources ",
    "Marketing": "Other",
    "Technical Degree": "Technical Degree",
    "Other": "Other",
    "Medical": "Other"
}

jrg_code = {
    "Healthcare Representative": "Other",
    "Laboratory Technician": "Other",
    "Human Resources": "Other",
    "Manager": "Other",
    "Manufacturing Director": "Manufacturing Director",
    "Research Director": "Research Director",
    "Research Scientist": "Other",
    "Sales Executive": "Other",
    "Sales Representative": "Other"
}

# Classe pour ajouter les features combinées (exactement comme dans ton code)
class CombinedAttributesAdder(object):
    def __init__(self):
        pass

    def transform(self, X):
        education_is_2 = (X['Education'] == 2).values
        job_level_is_2 = (X['JobLevel'] == 2).values
        job_involvement_is_1 = (X['JobInvolvement'] == 1).values
        environment_satisfaction_is_1 = (X['EnvironmentSatisfaction'] == 1).values
        work_life_balance_is_1 = (X['WorkLifeBalance'] == 1).values # Tu dois inclure WorkLifeBalance dans TOUTES_LES_COLONNES si tu l'utilises
        department_is_hr = (X['Department'] == "Human Resources").values

        def encoding_nparray(series, code):
            temp = pd.Series(series)
            return temp.map(code).values

        job_satisfaction_grouped = encoding_nparray(X["JobSatisfaction"].values, {1: 1, 2: 2, 3: 2, 4: 3})
        education_field_grouped = encoding_nparray(X["EducationField"].values, efg_code)
        job_role_grouped = encoding_nparray(X["JobRole"].values, jrg_code)

        single_and_travelling = ((X['MaritalStatus'] == 'Single') & (X["BusinessTravel"] != "Non-Travel")).values
        married_once_and_travelling = ((X['MaritalStatus'] != 'Single') & (X["BusinessTravel"] != "Non-Travel")).values
        do_overtime = (X['meanPresenceTime'] > 8).values
        is_young_and_worked_in_many_companies = ((X['Age'] <= 35) & (X['NumCompaniesWorked'] > 3)).values
        is_young_and_travel = ((X['Age'] <= 35) & (X["BusinessTravel"] != "Non-Travel")).values

        return np.c_[
            education_is_2,
            job_level_is_2,
            job_involvement_is_1,
            environment_satisfaction_is_1,
            work_life_balance_is_1,
            department_is_hr,
            job_satisfaction_grouped,
            education_field_grouped,
            job_role_grouped,
            single_and_travelling,
            married_once_and_travelling,
            do_overtime,
            is_young_and_worked_in_many_companies,
            is_young_and_travel,
        ]

# Fonction pour prétraiter les données entrées par l'utilisateur
def preprocess_user_input(data):
    df = pd.DataFrame([data], columns=TOUTES_LES_COLONNES)

    # Appliquer les mêmes transformations que dans ton pipeline
    cont_data = df[COLONNES_CONTINUES]
    disc_data = df[COLONNES_CATEGORIELLES]
    feat_eng_data = df[TOUTES_LES_COLONNES] # Applique l'ingénierie des features sur toutes les colonnes nécessaires

    # Imputation (si tu l'avais dans ton pipeline continu) - adapte le nombre de voisins si nécessaire
    imputer = joblib.load('imputer.joblib') # Assure-toi d'avoir sauvegardé ton imputer séparément si tu l'utilisais
    cont_processed = imputer.transform(cont_data)
    cont_processed_df = pd.DataFrame(cont_processed, columns=COLONNES_CONTINUES)

    # Encodage one-hot pour les variables discrètes
    encoder_disc = joblib.load('encoder_disc.joblib') # Assure-toi d'avoir sauvegardé ton encoder one-hot
    disc_processed = encoder_disc.transform(disc_data)
    disc_processed_df = pd.DataFrame(disc_processed) # Les noms de colonnes seront numériques après l'encodage

    # Ingénierie des features
    attribs_adder = CombinedAttributesAdder()
    feat_eng_processed = attribs_adder.transform(feat_eng_data)
    feat_eng_processed_df = pd.DataFrame(feat_eng_processed) # Les noms de colonnes seront numériques

    # Standardisation (si tu l'avais dans ton pipeline continu)
    scaler = joblib.load('scaler.joblib') # Assure-toi d'avoir sauvegardé ton scaler
    cont_scaled = scaler.transform(cont_processed_df)
    cont_scaled_df = pd.DataFrame(cont_scaled, columns=COLONNES_CONTINUES)

    # Combine les features prétraitées (l'ordre est important et doit correspondre à l'entraînement)
    processed_data = pd.concat([disc_processed_df, cont_scaled_df, feat_eng_processed_df], axis=1) # Adapte l'ordre en fonction de ton pipeline

    return processed_data

# Chargement du modèle
try:
    model = joblib.load('modele_attrition.joblib')
    print("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.title("Prédiction de l'attrition des employés")
st.write("Veuillez entrer les informations de l'employé pour prédire le risque d'attrition.")

# Création des widgets pour l'entrée utilisateur (adapte-les aux colonnes de ton DataFrame)
input_data = {}
for col in COLONNES_CATEGORIELLES:
    unique_values = hr_df[col].unique().tolist() # Récupère les valeurs uniques de ton DataFrame original
    input_data[col] = st.selectbox(f"Sélectionnez {col}", unique_values)

for col in COLONNES_CONTINUES:
    min_val = float(hr_df[col].min())
    max_val = float(hr_df[col].max())
    input_data[col] = st.number_input(f"Entrez {col}", min_value=min_val, max_value=max_val)

# Ajoute des widgets pour les autres colonnes nécessaires à l'ingénierie des features
input_data['Education'] = st.slider("Niveau d'éducation", 1, 5, 3)
input_data['JobLevel'] = st.slider("Niveau de poste", 1, 5, 2)
input_data['JobInvolvement'] = st.slider("Implication au travail", 1, 4, 3)
input_data['Department'] = st.selectbox("Département", hr_df['Department'].unique().tolist())
input_data['JobSatisfaction'] = st.slider("Satisfaction au travail", 1, 4, 3)
input_data['JobRole'] = st.selectbox("Rôle", hr_df['JobRole'].unique().tolist())
input_data['NumCompaniesWorked'] = st.slider("Nombre d'entreprises où il a travaillé", 0, 9, 1)
input_data['WorkLifeBalance'] = st.slider("Équilibre vie privée/vie pro", 1, 4, 3) # Assure-toi que cette colonne existe dans ton df original

if st.button("Prédire l'attrition"):
    try:
        processed_input = preprocess_user_input(input_data)
        prediction_proba = model.predict_proba(processed_input)[0][1] # Probabilité d'attrition (classe 'Yes' qui est encodée comme True ou 1)
        prediction_seuil = 0.5 # Tu peux ajuster le seuil si nécessaire
        prediction = "Oui" if prediction_proba >= prediction_seuil else "Non"

        st.subheader("Résultat de la prédiction")
        st.write(f"Probabilité d'attrition : {prediction_proba:.2f}")
        st.write(f"Prédiction d'attrition (seuil > {prediction_seuil}) : **{prediction}**")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
