# Importer des librairies
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import streamlit as st


# Configurer Streamlit
st.set_page_config(page_title = "Electric Vehicles Analysis", layout = "wide")


# Définir des fonctions
def calculate_charging_points_kpis(dataframe):
    """
    Fonction : Calculer des KPIs (nombre de ventes et taux d'évolution) concernant les bornes de recharge par année
    Résultat: Un dictionnaire concernant des KPIs (nombre de ventes et taux d'évolution) concernant les bornes de recharge par année
    """
    kpis_dict = {}    
    years = dataframe["year"].unique()
    for year in years:
        try:
            charging_points_number = dataframe.loc[
                (dataframe["unit"] == "charging points") & (dataframe["year"] == year)
            ]
            charging_points_total_number = int(charging_points_number["value"].sum())
            
            fast_charging_points_number = int(charging_points_number.loc[
                charging_points_number["powertrain"] == "Publicly available fast", "value"
            ].sum())
            slow_charging_points_number = int(charging_points_number.loc[
                charging_points_number["powertrain"] == "Publicly available slow", "value"
            ].sum())

            previous_year = year - 1
            charging_points_number_previous_year = dataframe.loc[
                (dataframe["unit"] == "charging points") & (dataframe["year"] == previous_year)
            ]
            charging_points_total_number_previous_year = int(charging_points_number_previous_year["value"].sum())

            if charging_points_total_number_previous_year > 0:
                evolution_rate = round((
                    ((charging_points_total_number - charging_points_total_number_previous_year) / charging_points_total_number_previous_year) * 100
                ), 2)
                
            else:
                evolution_rate = float("inf")
            
            kpis_dict[year] = {
                "total_charging_points": charging_points_total_number,
                "fast_charging_points": fast_charging_points_number,
                "slow_charging_points": slow_charging_points_number,
                "evolution_rate": evolution_rate
            }
        
        except Exception as e:
            print(f"Error message: {e}")

    return kpis_dict


def calculate_vehicles_kpis(dataframe, vehicle):
    """
    Fonction : Calculer des KPIs (nombre de ventes et taux d'évolution) concernant les véhicules électriques par année
    Résultat: Un dictionnaire concernant des KPIs (nombre de ventes et taux d'évolution) concernant les véhicules électriques par année
    """
    vehicles_kpis_dict = {}
    years = dataframe["year"].unique()
        
    for year in years:
        try:
            vehicle_sales_count = dataframe.loc[
                (dataframe["mode"] == vehicle) & 
                (dataframe["year"] == year) & 
                (dataframe["parameter"] == "EV sales")
            ]
            vehicle_total_sales_count = int(vehicle_sales_count["value"].sum())
    
            sales_count_for_BEV = int(vehicle_sales_count.loc[
                vehicle_sales_count["powertrain"] == "BEV", "value"
            ].sum())
                
            sales_count_for_PHEV = int(vehicle_sales_count.loc[
                vehicle_sales_count["powertrain"] == "PHEV", "value"
            ].sum())
                
            sales_count_for_FCEV = int(vehicle_sales_count.loc[
                vehicle_sales_count["powertrain"] == "FCEV", "value"
            ].sum())                    

            previous_year = year - 1
            vehicle_sales_count_previous_year = dataframe.loc[
                (dataframe["mode"] == vehicle) & 
                (dataframe["year"] == previous_year) & 
                (dataframe["parameter"] == "EV sales")
            ]
            vehicle_total_sales_count_previous_year = int(vehicle_sales_count_previous_year["value"].sum())

            if vehicle_total_sales_count_previous_year > 0:
                evolution_rate_total = round((
                    ((vehicle_total_sales_count - vehicle_total_sales_count_previous_year) / vehicle_total_sales_count_previous_year) * 100
                ), 2)
                
            else:
                evolution_rate_total = float("inf")

            previous_year_BEV_sales = int(vehicle_sales_count_previous_year.loc[
                vehicle_sales_count_previous_year["powertrain"] == "BEV", "value"
            ].sum())
            
            if previous_year_BEV_sales > 0:
                evolution_rate_BEV = round((
                    ((sales_count_for_BEV - previous_year_BEV_sales) / previous_year_BEV_sales) * 100
                ), 2)
                
            else:
                evolution_rate_BEV = float("inf")

            previous_year_PHEV_sales = int(vehicle_sales_count_previous_year.loc[
                vehicle_sales_count_previous_year["powertrain"] == "PHEV", "value"
            ].sum())
            
            if previous_year_PHEV_sales > 0:
                evolution_rate_PHEV = round((
                    ((sales_count_for_PHEV - previous_year_PHEV_sales) / previous_year_PHEV_sales) * 100
                ), 2)
            else:
                evolution_rate_PHEV = float("inf")

            previous_year_FCEV_sales = int(vehicle_sales_count_previous_year.loc[
                vehicle_sales_count_previous_year["powertrain"] == "FCEV", "value"
            ].sum())
            
            if previous_year_FCEV_sales > 0:
                evolution_rate_FCEV = round((
                    ((sales_count_for_FCEV - previous_year_FCEV_sales) / previous_year_FCEV_sales) * 100
                ), 2)
            else:
                evolution_rate_FCEV = float("inf")

            vehicles_kpis_dict[(vehicle, year)] = {
                "total_sales_count": vehicle_total_sales_count,
                "BEV_sales_count": sales_count_for_BEV,
                "PHEV_sales_count": sales_count_for_PHEV,
                "FCEV_sales_count": sales_count_for_FCEV,
                "evolution_rate_total": evolution_rate_total,
                "evolution_rate_BEV": evolution_rate_BEV,
                "evolution_rate_PHEV": evolution_rate_PHEV,
                "evolution_rate_FCEV": evolution_rate_FCEV
            }

        except Exception as e:
            print(f"Error processing {vehicle} in {year}: {e}")
            vehicles_kpis_dict[(vehicle, year)] = None

    return vehicles_kpis_dict


def train_and_predict(years_array, sales_counts, label):
    """
    Fonction : Entrainer un modèle de prédiction concernant le nombre de ventes pour l'année 2024
    Résultat: Un dictionnaire concernant la prédiction du nombre de ventes pour l'année 2024
    """
    polynomial_features = PolynomialFeatures(degree = 2)
    model = make_pipeline(polynomial_features, LinearRegression())
    model.fit(years_array, sales_counts)
    
    year_to_predict = np.array([[2024]])
    predicted_sales_count = model.predict(year_to_predict)
    
    return {label: predicted_sales_count[0]}


def calculate_kpis_and_predictions_for_region(region):
    """
    Fonction : Calculer des KPIs (nombre de ventes et taux d'évolution) concernant les véhicules électriques par région
    Résultat: Un dictionnaire concernant des KPIs (nombre de ventes et taux d'évolution) concernant les véhicules électriques par région
    """
    electric_vehicles_in_region_df = region_dfs_dict[region]
    
    # Calculer le nombre de ventes pour chaque type de véhicules
    cars_kpis = calculate_vehicles_kpis(electric_vehicles_in_region_df, "Cars")
    buses_kpis = calculate_vehicles_kpis(electric_vehicles_in_region_df, "Buses")
    trucks_kpis = calculate_vehicles_kpis(electric_vehicles_in_region_df, "Trucks")
    vans_kpis = calculate_vehicles_kpis(electric_vehicles_in_region_df, "Vans")

    # Extraire les données pour chaque type de véhicules 
    def extract_data(kpis, vehicle_type):
        years = [int(key[1]) for key in kpis.keys()]
        sales_count_all = [kpis[(vehicle_type, year)]["total_sales_count"] for year in years]
        sales_count_BEV = [kpis[(vehicle_type, year)]["BEV_sales_count"] for year in years]
        sales_count_PHEV = [kpis[(vehicle_type, year)]["PHEV_sales_count"] for year in years]
        sales_count_FCEV = [kpis[(vehicle_type, year)]["FCEV_sales_count"] for year in years]
        return years, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV

    data = {}
    for category in ["Cars", "Buses", "Trucks", "Vans"]:
        kpis = calculate_vehicles_kpis(electric_vehicles_in_region_df, category)
        data[category] = extract_data(kpis, category)
        
    # Prédire le nombre de ventes par type de véhicules
    def predict_sales(years, sales_count):
        years_array = np.array(years).reshape(-1, 1)
        sales_count_array = np.array(sales_count)
        prediction_dict = train_and_predict(years_array, sales_count_array, "total")
        return int(prediction_dict["total"])
        
    predictions = {}
    for category in ["Cars", "Buses", "Trucks", "Vans"]:
        years, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV = data[category]
        predictions[category] = {
            "sales_count_prediction_total": predict_sales(years, sales_count_all),
            "sales_count_prediction_BEV": predict_sales(years, sales_count_BEV),
            "sales_count_prediction_PHEV": predict_sales(years, sales_count_PHEV),
            "sales_count_prediction_FCEV": predict_sales(years, sales_count_FCEV)
        }
        
    return data, predictions


def plot_sales(years, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV, title):
    """
    Fonction : Afficher des graphiques sur Streamlit
    Résultat : Affichage de graphiques sur Streamlit
    """
    plt.figure(figsize = (15, 7.5))
    plt.plot(years,
             sales_count_all, 
             label = "All Electric Vehicles (BEVs, PHEVs and FCEVs)",
             color = "blue",
             marker = "o")
    plt.plot(years, 
             sales_count_BEV, 
             label = "Battery Electric Vehicles (BEVs)",
             color = "green", 
             marker = "s")
    plt.plot(years,
             sales_count_PHEV, 
             label = "Plug-in Hybrid Electric Vehicles (PHEVs)", 
             color = "magenta",
             marker = "^")
    plt.plot(years,
             sales_count_FCEV,
             label = "Fuel Cell Electric Vehicles (FCEVs)",
             color = "orange", 
             marker = "v")
    plt.legend(loc = "upper left")
    plt.xticks(years)
    plt.grid(linestyle = "--", alpha = 0.5)
    plt.xlabel("Year", fontsize = 12.5, fontstyle = "italic")
    plt.ylabel("Number of Sales", fontsize = 12.5, fontstyle = "italic")
    plt.title(title, fontsize = 15, fontweight = "bold")
    st.pyplot(plt)
    

# Stocker dans des variables les chemins d'accès des fichiers contenant les données
charging_points_file_path = r"C:\Users\willi\data_resources/portfolio/business_case/IEA_EV_dataEV_charging_pointsHistoricalEV.csv"
car_sales_file_path = r"C:\Users\willi\data_resources/portfolio/business_case/IEA_EV_dataEV_salesHistoricalCars.csv"
bus_sales_file_path = r"C:\Users\willi\data_resources/portfolio/business_case/IEA_EV_dataEV_stockHistoricalBuses.csv"
truck_sales_file_path = r"C:\Users\willi\data_resources/portfolio/business_case/IEA_EV_dataEV_stockHistoricalTrucks.csv"
van_sales_file_path = r"C:\Users\willi\data_resources/portfolio/business_case/IEA_EV_dataEV_stockHistoricalVans.csv"

# Lire les fichiers et Stocker leurs données dans des dataframes 
charging_points_df = pd.read_csv(charging_points_file_path)
car_sales_df = pd.read_csv(car_sales_file_path)
bus_sales_df = pd.read_csv(bus_sales_file_path)
truck_sales_df = pd.read_csv(truck_sales_file_path)
van_sales_df = pd.read_csv(van_sales_file_path)

# Concaténer les dataframes et Appeler la fonction check_dataframe() 
# pour afficher des informations concernant le dataframe
electric_vehicles_df = pd.concat([charging_points_df, car_sales_df, bus_sales_df, truck_sales_df, van_sales_df])

# Pour chaque région, Créer un dataframe et le Stocker dans un dictionnaire
region_dfs_dict = {}
regions = electric_vehicles_df["region"].unique()
for region in regions:
    region_dfs_dict[region] = electric_vehicles_df[electric_vehicles_df["region"] == region]


# Afficher un bloc de navigation avec des valeurs par défaut
st.sidebar.image(r"C:\Users\willi\data_resources/portfolio/business_case/business_case_logo.png", use_column_width = True)
st.sidebar.header("Navigation")
regions_for_navigation = sorted(electric_vehicles_df["region"].unique())
default_region_for_navigation = "World"
region_for_navigation = st.sidebar.selectbox(
    "Select a country",
    options = regions_for_navigation,
    index = regions_for_navigation.index(default_region_for_navigation)
)

default_year_for_navigation = 2023
years_for_navigation = sorted(electric_vehicles_df["year"].unique())
year_for_navigation = int(
    st.sidebar.selectbox(
        "Select a year",
        options = years_for_navigation,
        index = years_for_navigation.index(default_year_for_navigation)
    )
)
if year_for_navigation in years_for_navigation:
    year_index = years_for_navigation.index(year_for_navigation)
else:
    year_index = 0
    
default_vehicle_type_for_navigation = "Cars"
vehicle_types_for_navigation = ["Cars", "Buses", "Trucks", "Vans"]
vehicle_type_for_navigation = st.sidebar.selectbox(
    "Select a vehicle type",
    options = vehicle_types_for_navigation,
    index = vehicle_types_for_navigation.index(default_vehicle_type_for_navigation)
)

# Stocker les données filtrées par l'utilisateur dans un dataframe
user_selection_df = electric_vehicles_df[(electric_vehicles_df["region"] == region_for_navigation) & (electric_vehicles_df["year"] == year_for_navigation)]

# Utiliser la fonction calculate_kpis_and_predictions_for_region()
# et Stocker les résultats dans 2 dictionnaires
data, predictions = calculate_kpis_and_predictions_for_region(region_for_navigation)

# Extraire les données pour le type de véhicules sélectionné
years, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV = data[vehicle_type_for_navigation]
sales_count_prediction = predictions[vehicle_type_for_navigation]

# Afficher un titre et un sous-titre
centered_upper_text = f"""
<div style="text-align: center;">
    <h1 style="text-align: center;">VOLT MOTION INNOVATIONS</h1>
    <h2 style="text-align: center;">Dashboard of Electric Vehicle Sales</h2>
</div>
<br>
<br>
</div>
"""
st.markdown(centered_upper_text, unsafe_allow_html = True)

# Afficher les KPIs en fonction du type de véhicules sélectionné
centered_lower_text = f"""
<div style = "display: flex; justify-content: space-between; padding: 20px; text-align: center;">
    <div style = "flex: 1; margin-right: 20px;">
        <p style = "font-size: 25px;"><strong>{vehicle_type_for_navigation} Sales in {region_for_navigation} in {year_for_navigation}</strong></p>
        <p style = "font-size: 20px;">All Electric Vehicles (BEVs, PHEVs and FCEVs): {sales_count_all[year_index]}</p>
        <p style = "font-size: 20px;">Battery Electric Vehicles (BEVs): {sales_count_BEV[year_index]}</p>
        <p style = "font-size: 20px;">Plug-in Hybrid Electric Vehicles (PHEVs): {sales_count_PHEV[year_index]}</p>
        <p style = "font-size: 20px;">Fuel Cell Electric Vehicles (FCEVs): {sales_count_FCEV[year_index]}</p>
    </div>
    <div style = "flex: 1; margin-left: 20px; text-align: center;">
        <p style = "font-size: 25px;"><strong>{vehicle_type_for_navigation} Sales Predictions in {region_for_navigation} for 2024</strong></p>
        <p style = "font-size: 20px;">All Electric Vehicles (BEVs, PHEVs and FCEVs): {sales_count_prediction["sales_count_prediction_total"]}</p>
        <p style = "font-size: 20px;">Battery Electric Vehicles (BEVs): {sales_count_prediction["sales_count_prediction_BEV"]}</p>
        <p style = "font-size: 20px;">Plug-in Hybrid Electric Vehicles (PHEVs): {sales_count_prediction["sales_count_prediction_PHEV"]}</p>
        <p style = "font-size: 20px;">Fuel Cell Electric Vehicles (FCEVs): {sales_count_prediction["sales_count_prediction_FCEV"]}</p>
    </div>
</div>
<br>
<br>
<div style = "text-align: center;">
    <p style = "font-size: 25px;"><strong>Sales Chart</strong></p>
</div>
"""
st.markdown(centered_lower_text, unsafe_allow_html = True)

# Afficher les graphiques en fonction du type de véhicules sélectionné
if vehicle_type_for_navigation == "Cars":
    plot_sales(years_for_navigation, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV,
               "Evolution of the Number of Sales of Electric Cars (in Millions) from 2010 to 2023")
elif vehicle_type_for_navigation == "Buses":
    plot_sales(years_for_navigation, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV,
               "Evolution of the Number of Sales of Electric Buses (in Millions) from 2010 to 2023")
elif vehicle_type_for_navigation == "Trucks":
    plot_sales(years_for_navigation, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV,
               "Evolution of the Number of Sales of Electric Trucks (in Millions) from 2010 to 2023")
elif vehicle_type_for_navigation == "Vans":
    plot_sales(years_for_navigation, sales_count_all, sales_count_BEV, sales_count_PHEV, sales_count_FCEV,
               "Evolution of the Number of Sales of Electric Vans (in Millions) from 2010 to 2023")