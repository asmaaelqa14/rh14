import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Liste des colonnes nécessaires pour le modèle (à adapter précisément)
COLONNES_CATEGORIELLES = ["EducationField", "BusinessTravel", "EnvironmentSatisfaction", "MaritalStatus"]
COLONNES_CONTINUES = ["Age", "TotalWorkingYears", "YearsAtCompany", "YearsWithCurrManager", "meanPresenceTime"]
TOUTES_LES_COLONNES = COLONNES_CATEGORIELLES + COLONNES_CONTINUES + ['Education', 'JobLevel', 'JobInvolvement', 'Department', 'JobSatisfaction', 'JobRole', 'NumCompaniesWorked', 'WorkLifeBalance'] # Ajoute toutes les colonnes utilisées

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
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        education_is_2 = (X['Education'] == 2).values
        job_level_is_2 = (X['JobLevel'] == 2).values
        job_involvement_is_1 = (X['JobInvolvement'] == 1).values
        environment_satisfaction_is_1 = (X['EnvironmentSatisfaction'] == 1).values
        work_life_balance_is_1 = (X['WorkLifeBalance'] == 1).values
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

    cont_data = df[COLONNES_CONTINUES].copy()
    disc_data = df[COLONNES_CATEGORIELLES].copy()
    feat_eng_data = df[TOUTES_LES_COLONNES].copy()

    # Imputation
    try:
        imputer = joblib.load('imputer.joblib')
        cont_processed = imputer.transform(cont_data)
        cont_processed_df = pd.DataFrame(cont_processed, columns=COLONNES_CONTINUES)
    except FileNotFoundError:
        st.error("Le fichier imputer.joblib est introuvable. Assurez-vous de l'avoir téléchargé sur GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'imputer : {e}")
        st.stop()

    # Encodage one-hot pour les variables discrètes
    try:
        encoder_disc = joblib.load('encoder_disc.joblib')
        disc_processed = encoder_disc.transform(disc_data)
        disc_processed_df = pd.DataFrame(disc_processed, columns=encoder_disc.get_feature_names_out(COLONNES_CATEGORIELLES))
    except FileNotFoundError:
        st.error("Le fichier encoder_disc.joblib est introuvable. Assurez-vous de l'avoir téléchargé sur GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'encodeur discret : {e}")
        st.stop()

    # Ingénierie des features
    attribs_adder = CombinedAttributesAdder()
    feat_eng_processed = attribs_adder.transform(feat_eng_data)
    feat_eng_processed_df = pd.DataFrame(feat_eng_processed, columns=[
        'EducationIs2', 'JobLevelIs2', 'JobInvolvementIs1', 'EnvironmentSatisfactionIs1',
        'WorkLifeBalanceIs1', 'DepartmentIsHumanResources', 'JobSatisfactionGrouped',
        'EducationFieldGrouped', 'JobRoleGrouped', 'SingleAndTravelling',
        'MarriedOnceAndTravelling', 'DoOvertime', 'IsYoungAndWorkedInManyCompanies',
        'IsYoungAndTravel'
    ])

    # Standardisation
    try:
        scaler = joblib.load('scaler.joblib')
        cont_scaled = scaler.transform(cont_processed_df)
        cont_scaled_df = pd.DataFrame(cont_scaled, columns=COLONNES_CONTINUES)
    except FileNotFoundError:
        st.error("Le fichier scaler.joblib est introuvable. Assurez-vous de l'avoir téléchargé sur GitHub.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur lors du chargement du scaler : {e}")
        st.stop()

    # Combine les features prétraitées (l'ordre doit correspondre à l'entraînement)
    processed_data = pd.concat([disc_processed_df, cont_scaled_df, feat_eng_processed_df], axis=1)

    return processed_data

# Chargement du modèle
try:
    model = joblib.load('modele_attrition.joblib')
    print("Modèle chargé avec succès!")
except FileNotFoundError:
    st.error("Le fichier modele_attrition.joblib est introuvable. Assurez-vous de l'avoir téléchargé sur GitHub.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

st.title("Prédiction de l'attrition des employés")
st.write("Veuillez entrer les informations de l'employé pour prédire le risque d'attrition.")

# Création des widgets pour l'entrée utilisateur (adapte-les aux colonnes de ton DataFrame original 'hr_df')
# Il est important d'utiliser les valeurs uniques de ton DataFrame original pour les selectbox
try:
    hr_df_loaded = joblib.load('hr_df_for_streamlit_values.joblib') # Charge un petit extrait de ton hr_df pour les valeurs uniques
except FileNotFoundError:
    st.error("Le fichier hr_df_for_streamlit_values.joblib est introuvable. Veuillez le créer et le télécharger.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement des valeurs pour les widgets : {e}")
    st.stop()

input_data = {}
# Mise à jour des selectbox avec les valeurs uniques exactes
input_data["EducationField"] = st.selectbox("Sélectionnez EducationField", ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
input_data["BusinessTravel"] = st.selectbox("Sélectionnez BusinessTravel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
input_data["EnvironmentSatisfaction"] = st.selectbox("Sélectionnez EnvironmentSatisfaction", [ 3.,  2.,  4.,  1., float('nan')]) # Inclure NaN si présent
input_data["MaritalStatus"] = st.selectbox("Sélectionnez MaritalStatus", ['Married', 'Single', 'Divorced'])

for col in COLONNES_CONTINUES:
    min_val = float(hr_df_loaded[col].min())
    max_val = float(hr_df_loaded[col].max())
    input_data[col] = st.number_input(f"Entrez {col}", min_value=min_val, max_value=max_val)

input_data['Education'] = st.slider("Niveau d'éducation", 1, 5, int(hr_df_loaded['Education'].mode()[0]) if 'Education' in hr_df_loaded else 3)
input_data['JobLevel'] = st.slider("Niveau de poste", 1, 5, int(hr_df_loaded['JobLevel'].mode()[0]) if 'JobLevel' in hr_df_loaded else 2)
input_data['JobInvolvement'] = st.slider("Implication au travail", 1, 4, int(hr_df_loaded['JobInvolvement'].mode()[0]) if 'JobInvolvement' in hr_df_loaded else 3)
input_data['Department'] = st.selectbox("Département", ['Sales', 'Research & Development', 'Human Resources'])
input_data['JobSatisfaction'] = st.slider("Satisfaction au travail", 1, 4, int(hr_df_loaded['JobSatisfaction'].mode()[0]) if 'JobSatisfaction' in hr_df_loaded else 3)
input_data['JobRole'] = st.selectbox("Rôle", ['Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Human Resources', 'Research Director', 'Laboratory Technician', 'Manufacturing Director', 'Sales Representative', 'Manager'])
input_data['NumCompaniesWorked'] = st.slider("Nombre d'entreprises où il a travaillé", 0, int(hr_df_loaded['NumCompaniesWorked'].max()) if 'NumCompaniesWorked' in hr_df_loaded else 9, int(hr_df_loaded['NumCompaniesWorked'].mode()[0]) if 'NumCompaniesWorked' in hr_df_loaded else 1)
input_data['WorkLifeBalance'] = st.slider("Équilibre vie privée/vie pro", 1, 4, int(hr_df_loaded['WorkLifeBalance'].mode()[0]) if 'WorkLifeBalance' in hr_df_loaded else 3)
input_data['meanPresenceTime'] = st.number_input("Temps de présence moyen (en heures)", min_value=0.0, max_value=24.0, value=8.0)
input_data['MaritalStatus'] = st.selectbox("Statut marital", ['Married', 'Single', 'Divorced'])
input_data['BusinessTravel'] = st.selectbox("Fréquence des voyages d'affaires", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
input_data['Age'] = st.number_input("Âge", min_value=18, max_value=100, value=30)

if st.button("Prédire l'attrition"):
    try:
        processed_input = preprocess_user_input(input_data)
        prediction_proba = model.predict_proba(processed_input)[0][1] # Probabilité d'attrition (classe 'Yes')
        prediction_seuil = 0.5
        prediction = "Oui" if prediction_proba >= prediction_seuil else "Non"

        st.subheader("Résultat de la prédiction")
        st.write(f"Probabilité d'attrition : {prediction_proba:.2f}")
        st.write(f"Prédiction d'attrition (seuil > {prediction_seuil}) : **{prediction}**")

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# Pour obtenir les valeurs uniques pour les selectbox, tu peux sauvegarder un petit extrait de ton hr_df
# dans Google Colab et le télécharger également :
# hr_df[['EducationField', 'BusinessTravel', 'EnvironmentSatisfaction', 'MaritalStatus', 'Department', 'JobRole']].to_pickle('hr_df_for_streamlit_values.pkl')
# Puis, dans ton app.py, tu peux charger ce fichier :
# hr_df_loaded = pd.read_pickle('hr_df_for_streamlit_values.pkl')

# Alternative (moins recommandée pour la taille) :
# hr_df.to_pickle('hr_df_full.pkl')
# hr_df_loaded = pd.read_pickle('hr_df_full.pkl')
