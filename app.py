import streamlit as st
import pickle
import numpy as np

# Chargement du modèle entraîné
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Prédiction de la fidélité des employés")

st.write("Remplissez les informations ci-dessous :")

# Inputs utilisateur
age = st.number_input("Âge", min_value=18, max_value=60, value=30)
distance_from_home = st.number_input("Distance du domicile (km)", min_value=1, max_value=30, value=5)
education = st.selectbox("Niveau d'éducation", [1, 2, 3, 4, 5])
job_level = st.selectbox("Niveau de poste", [1, 2, 3, 4, 5])
monthly_income = st.number_input("Salaire mensuel", min_value=1000, max_value=200000, value=50000)
num_companies_worked = st.selectbox("Nombre d'entreprises précédentes", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
percent_salary_hike = st.selectbox("Augmentation de salaire (%)", [11, 12, 13, 14, 15, 17, 20, 21, 22, 23])
total_working_years = st.number_input("Années totales d'expérience", min_value=0, max_value=40, value=5)
training_times_last_year = st.selectbox("Formations suivies l'année dernière", [0, 1, 2, 3, 4, 5, 6])
years_at_company = st.number_input("Années dans l'entreprise", min_value=0, max_value=40, value=3)
years_since_last_promotion = st.selectbox("Années depuis la dernière promotion", [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])
years_with_curr_manager = st.selectbox("Années avec le manager actuel", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
job_involvement = st.selectbox("Implication au travail", [1, 2, 3, 4])
performance_rating = st.selectbox("Évaluation de performance", [3, 4])
environment_satisfaction = st.selectbox("Satisfaction environnementale", [1, 2, 3, 4])
job_satisfaction = st.selectbox("Satisfaction professionnelle", [1, 2, 3, 4])
work_life_balance = st.selectbox("Équilibre vie pro/perso", [1, 2, 3, 4])
stock_option_level = st.selectbox("Niveau d'options d'achat d'actions", [0, 1, 2, 3])
mean_start_time = st.number_input("Heure moyenne d'arrivée", min_value=5.0, max_value=12.0, value=9.9)
mean_leave_time = st.number_input("Heure moyenne de départ", min_value=13.0, max_value=22.0, value=17.0)
mean_presence_time = st.number_input("Temps de présence moyen", min_value=4.0, max_value=12.0, value=7.5)
number_of_absence_day = st.selectbox("Nombre de jours d'absence", list(range(0, 25)))
number_of_presence_day = st.selectbox("Nombre de jours de présence", list(range(200, 251)))

# Catégorielles
business_travel = st.selectbox("Fréquence de voyage professionnel", 
                               ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
department = st.selectbox("Département", 
                          ['Sales', 'Research & Development', 'Human Resources'])
education_field = st.selectbox("Domaine d'éducation", 
                               ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
gender = st.selectbox("Genre", ['Female', 'Male'])
job_role = st.selectbox("Poste", 
                        ['Healthcare Representative', 'Research Scientist', 'Sales Executive',
                         'Human Resources', 'Research Director', 'Laboratory Technician',
                         'Manufacturing Director', 'Sales Representative', 'Manager'])
marital_status = st.selectbox("État civil", ['Married', 'Single', 'Divorced'])

# Construction de l'entrée
input_data = {
    'Age': age,
    'DistanceFromHome': distance_from_home,
    'Education': education,
    'JobLevel': job_level,
    'MonthlyIncome': monthly_income,
    'NumCompaniesWorked': num_companies_worked,
    'PercentSalaryHike': percent_salary_hike,
    'TotalWorkingYears': total_working_years,
    'TrainingTimesLastYear': training_times_last_year,
    'YearsAtCompany': years_at_company,
    'YearsSinceLastPromotion': years_since_last_promotion,
    'YearsWithCurrManager': years_with_curr_manager,
    'JobInvolvement': job_involvement,
    'PerformanceRating': performance_rating,
    'EnvironmentSatisfaction': environment_satisfaction,
    'JobSatisfaction': job_satisfaction,
    'WorkLifeBalance': work_life_balance,
    'StockOptionLevel': stock_option_level,
    'meanStartTime': mean_start_time,
    'meanLeaveTime': mean_leave_time,
    'meanPresenceTime': mean_presence_time,
    'numberOfAbsenceDay': number_of_absence_day,
    'numberOfPresenceDay': number_of_presence_day,
    'BusinessTravel': business_travel,
    'Department': department,
    'EducationField': education_field,
    'Gender': gender,
    'JobRole': job_role,
    'MaritalStatus': marital_status
}

# Encodage si pipeline avec OneHotEncoder ou autre
import pandas as pd
input_df = pd.DataFrame([input_data])

if st.button("Prédire"):
    prediction = model.predict(input_df)
    st.success(f"Résultat de la prédiction : {prediction[0]}")

