from flask import Blueprint, render_template, request
from .data_handler import (
    get_unique_countries, plot_avg, compare_gdp_with_wdi, get_unique_WDI,
    plot_top_20, get_unique_years, get_wdi_year_country, plot_skewness,
    corr_matrix_heatmap, predict_gdp, predict_gdp_with_model
)

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    overviews = ['Show skewness', 'Show correlation heatmap']
    selected_overview = None
    countries = get_unique_countries()
    wdis = get_unique_WDI()
    years = get_unique_years()
    graph_html = None
    error_message = None
    selected_country = None
    selected_wdi = None
    selected_year = 2000
    value_message = None
    action = None
    wdi_values = None
    if request.method == 'POST':
        action = request.form.get('action')

        if action in ["Show Avg", "Compare GDP with WDI", "Top 20", "Get Value", "Overview", "Predict"]:
            graph_html = None  # Reset the graph
            value_message = None  # Reset value message
            error_message = None  # Reset error message

        selected_country = request.form.get('country')
        selected_wdi = request.form.get('wdi')
        selected_year = int(request.form.get('year', 2000)) if request.form.get('year') is not None else 2000
        selected_overview = request.form.get('overview')

        if action == "Show Avg":
            if selected_wdi:
                graph_html = plot_avg(selected_country, selected_wdi)
                #graph_html = None
            else:
                error_message = "Please select a World Development Indicator."
        elif action == "Compare GDP with WDI":
            if selected_wdi:
                graph_html = compare_gdp_with_wdi(selected_country, selected_wdi)
            else:
                error_message = "Please select a World Development Indicator."
        elif action == "Top 20":
            if selected_year and selected_wdi:
                graph_html = plot_top_20(selected_year, selected_wdi)
            else:
                error_message = "Please select a year and a World Development Indicator."
        elif action == "Get Value":
            value_message = get_wdi_year_country(selected_country, selected_wdi, selected_year)
        elif action == "Overview":
            if selected_overview == 'Show skewness':
                graph_html = plot_skewness()
            elif selected_overview == 'Show correlation heatmap':
                graph_html = corr_matrix_heatmap()
        elif action == "Predict":
            try:
                wdi_values = {
                    "clean_fuels": request.form.get("clean_fuels"),
                    "electricity": request.form.get("electricity"),
                    "alcohol_consumption": request.form.get("alcohol_consumption"),
                    "co2_emissions": request.form.get("co2_emissions"),
                    "health_expenditure": request.form.get("health_expenditure"),
                    "immunization_dpt": request.form.get("immunization_dpt"),
                    "immunization_hepb": request.form.get("immunization_hepb"),
                    "immunization_measles": request.form.get("immunization_measles"),
                    "life_expectancy": request.form.get("life_expectancy"),
                    "hospital_beds": request.form.get("hospital_beds"),
                    "mortality_cvd": request.form.get("mortality_cvd"),
                    "mortality_rate_female": request.form.get("mortality_rate_female"),
                    "mortality_rate_male": request.form.get("mortality_rate_male"),
                    "mortality_road_injury": request.form.get("mortality_road_injury"),
                    "mortality_under_5": request.form.get("mortality_under_5"),
                    "physicians": request.form.get("physicians"),
                    "pm25_exposure": request.form.get("pm25_exposure"),
                    "population": request.form.get("population"),
                    "undernourishment": request.form.get("undernourishment"),
                    "suicide_mortality": request.form.get("suicide_mortality"),
                    "unemployment": request.form.get("unemployment"),
                }
                
                # Convert string inputs to floats where applicable; keep None for empty inputs
                for key in wdi_values:
                    if wdi_values[key] and wdi_values[key].strip() != '':
                        wdi_values[key] = float(wdi_values[key])
                    else:
                        wdi_values[key] = None  # Allow interpolation to handle missing values

                # Predict GDP
                predicted_gdp = predict_gdp(selected_year, selected_country, **wdi_values)
                value_message = f"Predicted GDP for {selected_country} in {selected_year} is: ${predicted_gdp:,.2f}"
            except Exception as e:
                error_message = str(e)

    return render_template('index.html',
                           wdi_values=wdi_values,
                           overviews=overviews, 
                           selected_overview=selected_overview, 
                           value_message=value_message, 
                           years=years, 
                           countries=countries, 
                           wdis=wdis,
                           selected_wdi=selected_wdi, 
                           action=action, 
                           selected_year=int(selected_year), 
                           selected_country=selected_country, 
                           graph_html=graph_html, 
                           error_message=error_message)