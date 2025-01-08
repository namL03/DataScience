import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np
best_model = joblib.load('app/optimized_random_forest_gdp_model.pkl')
def load_data():
    # Load the dataset ensuring to handle any specific parsing needs
    df = pd.read_csv("app/data/World Bank Dataset.csv")
    return df

def get_unique_countries():
    df = load_data()
    return sorted(df['Country Name'].dropna().unique())

def get_unique_years():
    df = load_data()
    return sorted(df['Year'].dropna().unique())

def get_unique_WDI():
    df = load_data()
    wdi_columns = df.columns[2:]  
    return wdi_columns.tolist()

def plot_avg(country, wdi):
    # Filter data by country and plot average GDP
    df = load_data()
    df_filtered = df[df['Country Name'] == country]
    df_grouped = df_filtered.groupby('Year').agg({wdi: 'mean'}).reset_index()
    fig = px.line(df_grouped, x='Year', y=wdi, title=f'Changes of {wdi} in {country} throughout the years')
    return fig.to_html(full_html=False)

def plot_top_20(year, wdi):
    df = load_data()
    #print(year)
    df_filtered = df[df['Year'] == int(year)]
    #print(df_filtered.head())
    df_top20 = df_filtered.sort_values(by=wdi, ascending=False).head(20)
    fig = px.bar(df_top20, x='Country Name', y=wdi, title=f'Top 20 Countries by {wdi} in {year}',
                 labels={'Country Name': 'Country', wdi: f'{wdi} Value'})
    return fig.to_html(full_html=False)

def compare_gdp_with_wdi(country, wdi_select):
    df = load_data()
    df_filtered = df[df['Country Name'] == country]
    if wdi_select in df.columns:
        fig = px.scatter(df_filtered, x='GDP (current US$)', y=wdi_select,
                         title=f'GDP vs {wdi_select} in {country}', trendline="ols")
        return fig.to_html(full_html=False)
    else:
        return "<p>Selected WDI not found in the dataset.</p>"

def get_wdi_year_country(country, wdi, year):
    df = load_data()
    filtered_df = df[(df['Country Name'] == country) & (df['Year'] == int(year))]
    #print(filtered_df.head())
    return filtered_df[wdi].values[0]

import plotly.express as px

def corr_matrix_heatmap():
    df = load_data()
    # Filter to include only numerical columns for correlation matrix
    numerical_df = df.select_dtypes(include=['number'])
    corr_matrix = numerical_df.corr()
    
    # Generate a list of column indices (0-based)
    indices = list(range(len(numerical_df.columns)))
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    labels=dict(x="Variable Index", y="Variable Index", color="Correlation"),
                    title="Correlation Matrix Heatmap")

    # Update x and y axes to use indices instead of column names
    fig.update_xaxes(tickvals=indices, ticktext=indices)
    fig.update_yaxes(tickvals=indices, ticktext=indices)

    return fig.to_html(full_html=False)

def plot_skewness():
    df = load_data()
    # Calculate skewness for numerical columns
    skewness = df.select_dtypes(include=['number']).skew().reset_index()
    skewness.columns = ['Column', 'Skewness']

    # Plotting the skewness values using Plotly Express
    fig = px.bar(skewness, y='Column', x='Skewness', title="Skewness of Numerical Columns",
                 labels={'Column': 'Column', 'Skewness': 'Skewness Value'},
                 color='Skewness', color_continuous_scale='Viridis',
                 orientation='h')  # orientation 'h' ensures horizontal bar chart

    # Remove y-axis title
    fig.update_yaxes(title_text='')

    # Convert plot to HTML for embedding in web interfaces or Jupyter Notebooks
    return fig.to_html(full_html=False)


def predict_gdp(year, country, **wdi_values):
    import pandas as pd

    # Load the dataset
    df = load_data()

    # Validate the year
    if not isinstance(year, int) or year < 0:
        raise ValueError("Year must be a positive integer.")

    # Check if the specified country has data for 2018 and 2019
    df_filtered = df[(df['Country Name'] == country) & (df['Year'].isin([2018, 2019]))]

    if df_filtered.empty or len(df_filtered) < 2:
        raise ValueError(f"Not enough data available for {country} to perform prediction (requires 2018 and 2019 GDP).")

    # Extract GDP values for 2018 and 2019
    gdp_2018 = df_filtered[df_filtered['Year'] == 2018]['GDP (current US$)'].values[0]
    gdp_2019 = df_filtered[df_filtered['Year'] == 2019]['GDP (current US$)'].values[0]

    # Apply the formula for linear extrapolation
    printed_gdp = (gdp_2019 - gdp_2018) * (year - 2019) + gdp_2019

    # Debugging: Print calculation details
    print(f"GDP in 2018: {gdp_2018}, GDP in 2019: {gdp_2019}")
    print(f"Extrapolated GDP for {year}: {printed_gdp}")

    return printed_gdp

def predict_gdp_with_model(year, country, **wdi_values):
    # Load the dataset and model
    df = load_data()  # Assumes this function loads the dataset (historical WDI values with GDP)
    best_model = joblib.load('app/optimized_random_forest_gdp_model.pkl')

    # Validate the year
    if not isinstance(year, int) or year < 0:
        raise ValueError("Year must be a positive integer.")

    # Define the fixed order of features from the model
    fixed_feature_order = [
        "Access to clean fuels and technologies for cooking (% of population)",
        "Access to electricity (% of population)",
        "Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)",
        "CO2 emissions (metric tons per capita)",
        "Current health expenditure per capita (current US$)",
        "Immunization, DPT (% of children ages 12-23 months)",
        "Immunization, HepB3 (% of one-year-old children)",
        "Immunization, measles (% of children ages 12-23 months)",
        "Life expectancy at birth, total (years)",
        "Hospital beds (per 1,000 people)",
        "Mortality from CVD, cancer, diabetes or CRD between exact ages 30 and 70 (%)",
        "Mortality rate, adult, female (per 1,000 female adults)",
        "Mortality rate, adult, male (per 1,000 male adults)",
        "Mortality caused by road traffic injury (per 100,000 population)",
        "Mortality rate, under-5 (per 1,000 live births)",
        "Physicians (per 1,000 people)",
        "PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)",
        "Population, total",
        "Prevalence of undernourishment (% of population)",
        "Suicide mortality rate (per 100,000 population)",
        "Unemployment, total (% of total labor force) (modeled ILO estimate)"
    ]

    # Filter historical data for the specified country
    df_country = df[df['Country Name'] == country]

    if df_country.empty:
        raise ValueError(f"No historical data available for {country}.")

    # Initialize the WDI values for the specified year
    extrapolated_features = {}

    # Loop through each feature in the fixed_feature_order
    for feature in fixed_feature_order:
        # Check if the feature is provided in wdi_values
        if feature in wdi_values and wdi_values[feature] is not None:
            extrapolated_features[feature] = float(wdi_values[feature])
        else:
            # Perform linear extrapolation for missing features
            historical_data = df_country[['Year', feature]].dropna()

            # Ensure there are at least two data points for extrapolation
            if historical_data.shape[0] < 2:
                raise ValueError(f"Not enough historical data for '{feature}' to perform extrapolation.")

            # Get the most recent two years of data
            historical_data = historical_data.sort_values(by='Year')
            year_1, value_1 = historical_data.iloc[-2]['Year'], historical_data.iloc[-2][feature]
            year_2, value_2 = historical_data.iloc[-1]['Year'], historical_data.iloc[-1][feature]

            # Apply the linear extrapolation formula
            extrapolated_value = (value_2 - value_1) * (year - year_2) / (year_2 - year_1) + value_2
            extrapolated_features[feature] = extrapolated_value

    # Debugging: Print extrapolated features
    print(f"Extrapolated Features for {country} in {year}:")
    for k, v in extrapolated_features.items():
        print(f"{k}: {v}")

    # Arrange the features in the correct order for the model
    feature_values = [extrapolated_features[feature] for feature in fixed_feature_order]

    # Convert to numpy array for model input
    feature_values = np.array(feature_values).reshape(1, -1)

    # Predict GDP using the pre-trained model
    predicted_gdp = best_model.predict(feature_values)[0]

    # Debugging: Print the prepared features and prediction
    print("Prepared Features for Model:", feature_values)
    print("Predicted GDP:", predicted_gdp)

    return predicted_gdp