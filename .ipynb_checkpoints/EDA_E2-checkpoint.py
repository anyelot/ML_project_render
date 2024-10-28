import dash
from dash import dcc
from dash import html
import dash_table
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.tsa.api as smtsa
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from scipy.ndimage import uniform_filter1d
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import plotly.express as px
import pycountry
import plotly.graph_objects as go
from scipy.stats import kruskal
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

df_full = pd.read_csv("C:/Users/ADM/Documents/MachineLearningUN/ML_ProyectoFinal/base/owid-energy-data.csv", sep=",")

df_full.describe(include = object)

# Eliminamos filas que tengan NaN en las columnas 'country', 'year' o 'population'
df = df_full.dropna(subset=['country', 'year', 'population'])

# Mostramos las primeras filas del DataFrame resultante
df.head(9)

ren_energy_columns = [
    'biofuel_consumption', 
    'hydro_consumption', 
    'solar_consumption', 
    'wind_consumption', 
    'nuclear_consumption', 
    'other_renewable_consumption',
]

nonren_energy_columns = [
    'fossil_fuel_consumption',
    'coal_consumption',
    'gas_consumption',
    'oil_consumption'
]

df_filtered_range = df[(df['year'] >= 1990) & (df['year'] <= 2022)]
df_filtered_range = df_filtered_range[(df['country'] != 'World') & 
                (df_filtered_range['country'] != 'Europe') & 
                (df_filtered_range['country'] != 'Asia') & 
                (df_filtered_range['country'] != 'North America') & 
                (df_filtered_range['country'] != 'High-income countries') & 
                (df_filtered_range['country'] != 'Lower-middle-income countries') & 
                (df_filtered_range['country'] != 'Upper-middle-income countries') & 
                (df_filtered_range['country'] != 'European Union (27)') & 
                (df_filtered_range['country'] != 'South America')]

# Sumar energías para cada categoría
df_filtered_range['ren_energy'] = df_filtered_range[ren_energy_columns].sum(axis=1)
df_filtered_range['nonren_energy'] = df_filtered_range[nonren_energy_columns].sum(axis=1)

top10_countries = df_filtered_range.groupby('country')['ren_energy'].sum().nlargest(10).index
df_top10 = df_filtered_range[df_filtered_range['country'].isin(top10_countries)]

# Crear un boxplot por año mostrando la distribución de consumo de energías renovables por país
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_top10, x='year', y='ren_energy', palette='YlOrBr')

plt.title('Distribución del consumo de energías renovables por año (1990 - 2022)')
plt.xlabel('Año')
plt.ylabel('Consumo total de energías renovables (TWh)')

# Mostrar el gráfico
plt.tight_layout()

# 1. Eliminar valores atípicos utilizando el método del IQR
Q1 = df_top10['ren_energy'].quantile(0.25)
Q3 = df_top10['ren_energy'].quantile(0.75)
IQR = Q3 - Q1

# Definir el rango para detectar outliers
lower_bound = Q1 - 0.25 * IQR
upper_bound = Q3 + 0.25 * IQR

# Filtrar datos dentro del rango intercuartílico (eliminar outliers)
df_no_outliers = df_filtered_range[(df_filtered_range['ren_energy'] >= lower_bound) & 
                                   (df_filtered_range['ren_energy'] <= upper_bound)]

# Función para aplicar ARIMA a cada columna de energía
def apply_arima(series):
    # Asegurarse de que no haya valores faltantes en la serie
    series = series.dropna()
    
    # Si hay datos suficientes, aplicamos ARIMA
    if len(series) > 10:  # Esto es para evitar series muy cortas
        model = sm.tsa.ARIMA(series, order=(5,1,0))  # Order ARIMA(p,d,q)
        model_fit = model.fit()  # Aquí quitamos el argumento disp
        forecasted_values = model_fit.predict(start=0, end=len(series)-1, typ='levels')
    else:
        # Si la serie es muy corta, devolvemos los valores originales sin imputar
        forecasted_values = series

    return forecasted_values

# Aplicar ARIMA a cada columna de energías renovables
df_no_outliers[ren_energy_columns] = df_no_outliers[ren_energy_columns].apply(lambda col: apply_arima(col), axis=0)

# Visualizar el boxplot de energías renovables después de la imputación por ARIMA
plt.figure(figsize=(14, 8))
sns.boxplot(data=df_no_outliers, x='year', y='ren_energy', palette='YlOrBr')

plt.title('Consumo de energías renovables por año después de la imputación ARIMA (1990 - 2022)')
plt.xlabel('Año')
plt.ylabel('Consumo total de energías renovables (TWh)')
plt.tight_layout()

# Dibujar la serie temporal original e imputada de manera más clara
plt.figure(figsize=(14, 10))

for i, energy in enumerate(ren_energy_columns, 1):
    plt.subplot(3, 2, i)
    
    # Dibujar la serie original con líneas más suaves y sin marcadores
    plt.plot(df_filtered_range['year'], df_filtered_range[energy], label=f'{energy} - Original', color='blue', alpha=0.7)
    
    # Dibujar la serie después de la imputación ARIMA
    plt.plot(df_no_outliers['year'], df_no_outliers[energy], label=f'{energy} - Imputado ARIMA', color='orange', linestyle='--', alpha=0.8)
    
    # Configuraciones del gráfico
    plt.title(f'Serie de {energy}: Original vs Imputado')
    plt.xlabel('Año')
    plt.ylabel('Consumo (TWh)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.ylim([df_filtered_range[energy].min() - 10, df_filtered_range[energy].max() + 10])  # Ajustar eje Y para visualizar mejor

plt.tight_layout()

plt.figure(figsize=(14, 10))

for i, energy in enumerate(ren_energy_columns, 1):
    plt.subplot(3, 2, i)
    
    # Histograma de la serie original
    sns.histplot(df_filtered_range['ren_energy'].dropna(), color='blue', kde=True, label='Original', alpha=0.5, bins=30)
    
    # Histograma de la serie imputada
    sns.histplot(df_no_outliers['ren_energy'].dropna(), color='orange', kde=True, label='Imputado', alpha=0.5, bins=30)
    
    plt.title(f'Distribución de {energy}: Original vs Imputado')
    plt.xlabel('Consumo (TWh)')
    plt.ylabel('Frecuencia')
    plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Realizar la prueba de Kruskal-Wallis para cada tipo de energía
for energy in ren_energy_columns:
    stat, p_value = kruskal(df_filtered_range[energy].dropna(), df_no_outliers[energy].dropna())
    print(f'Prueba Kruskal-Wallis para {energy}: Estadístico = {stat:.3f}, valor p = {p_value:.3f}')

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1) 
df_filtered_plot = df_filtered_range[['year'] + ren_energy_columns].groupby('year').sum().reset_index()

for energy in ren_energy_columns:
    plt.plot(df_filtered_plot['year'], df_filtered_plot[energy], label=energy)

plt.title('Consumo de diferentes fuentes de energía renovables (1990 - 2022)')
plt.xlabel('Año')
plt.ylabel('Consumo (TWh)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)  
df_filtered_plot_nonren = df_filtered_range[['year'] + nonren_energy_columns].groupby('year').sum().reset_index()

for energy in nonren_energy_columns:
    plt.plot(df_filtered_plot_nonren['year'], df_filtered_plot_nonren[energy], label=energy)

plt.title('Consumo de diferentes fuentes de energía no renovables (1990 - 2022)')
plt.xlabel('Año')
plt.ylabel('Consumo (TWh)')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

df_total_energy_per_year = df_filtered_range.groupby('year')[['ren_energy', 'nonren_energy']].sum().reset_index()
plt.figure(figsize=(10, 8))

def plot_bars(ax, data, color, title, y_label):
    bars = ax.bar(data['year'], data['value'], color=color, width=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.05 * yval, f'{round(yval/1000, 1)}K', ha='center', va='bottom', rotation=90)
    ax.set_title(title)
    ax.set_xlabel('Año')
    ax.set_ylabel(y_label)
    ax.grid(axis='y', linestyle='--', linewidth=0.7)
    ax.set_ylim(0, yval + yval * 0.35)

plt.subplot(2, 1, 1)
plot_bars(plt.gca(), df_total_energy_per_year.rename(columns={'ren_energy': 'value'}), '#90EE90', 'Suma total de fuentes de energía renovable (1990 - 2022)', 'Consumo Total (TWh)')

plt.subplot(2, 1, 2)
plot_bars(plt.gca(), df_total_energy_per_year.rename(columns={'nonren_energy': 'value'}), '#FFA07A', 'Suma total de fuentes de energía no renovable (1990 - 2022)', 'Consumo Total (TWh)')

plt.tight_layout()

df_renewable_total = df_filtered_range[ren_energy_columns].sum()
df_nonrenewable_total = df_filtered_range[nonren_energy_columns].sum()

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.pie(df_renewable_total, autopct='%1.1f%%', startangle=90, 
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0','#ffb3e6'])
plt.title('Porcentaje del consumo total de energías renovables (1990 - 2022)')
plt.axis('equal')  
plt.legend(ren_energy_columns, loc="best", bbox_to_anchor=(1, 0, 0.5, 1))  

plt.subplot(1, 2, 2)
plt.pie(df_nonrenewable_total, autopct='%1.1f%%', startangle=90, 
        colors=['#ff6666', '#ffcc66', '#66ff66', '#66ccff'])
plt.title('Porcentaje del consumo total de energías no renovables (1990 - 2022)')
plt.axis('equal')  
plt.legend(nonren_energy_columns, loc="best", bbox_to_anchor=(1, 0, 0.5, 1)) 

plt.tight_layout()

df_renewable_consumption = df_filtered_range.groupby('country')[ren_energy_columns].sum().reset_index()

df_renewable_consumption['total_renewable'] = df_renewable_consumption[ren_energy_columns].sum(axis=1)

# Configuración para ocultar barra de herramientas
config = {'showLink': False, 'displayModeBar': False}

# Crear el mapa de coropletas con plotly
fig = px.choropleth(
    df_renewable_consumption,
    locations="country",  # Columna con los nombres de los países
    locationmode="country names",  # Especifica que los nombres corresponden a países
    color="total_renewable",  # Columna con los valores a colorear
    hover_name="country",  # Columna que se mostrará al pasar el mouse
    title="Consumo total de energías renovables por país (1990 - 2022)",  # Título del mapa
    color_continuous_scale=px.colors.sequential.YlGnBu,  # Escala de colores
    range_color=(0, 150000)  # Rango de los colores
)

# Actualizar el diseño
fig.update_layout(
    title_text='Consumo total de energías renovables por país (1990 - 2022)',
    geo=dict(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white"),
    coloraxis_colorbar=dict(title="Consumo Total de Energía (TWh)")
)

df_top10_renewable = df_renewable_consumption.sort_values(by='total_renewable', ascending=False).head(10)

# Crear la gráfica de barras
plt.figure(figsize=(10, 6))
plt.barh(df_top10_renewable['country'], df_top10_renewable['total_renewable'], color='orange')

# Añadir etiquetas y títulos
for index, value in enumerate(df_top10_renewable['total_renewable']):
    plt.text(value, index, f'{value:.2f} TWh')

plt.title('Top 10 países con mayor consumo de energías renovables (1990 - 2022)')
plt.xlabel('Consumo Total de Energía (TWh)')
plt.ylabel('País')

df_annual_consumption = df_filtered_range.groupby('year')[ren_energy_columns + nonren_energy_columns].sum().reset_index()

df_annual_consumption['total_renewable'] = df_annual_consumption[ren_energy_columns].sum(axis=1)
df_annual_consumption['total_nonrenewable'] = df_annual_consumption[nonren_energy_columns].sum(axis=1)

df_annual_consumption['percentage_renewable'] = (df_annual_consumption['total_renewable'] / 
                                                 (df_annual_consumption['total_renewable'] + df_annual_consumption['total_nonrenewable'])) * 100

fig = px.line(df_annual_consumption, 
              x='year', 
              y='percentage_renewable', 
              title='Porcentaje de consumo de energías renovables vs no renovables (1990 - 2022)',
              labels={'percentage_renewable': 'Porcentaje de Energías Renovables (%)', 'year': 'Año'},
              markers=True)

fig.update_traces(texttemplate='%{y:.1f}%', textposition="top right")

plt.figure(figsize=(8, 8))
ax = sns.heatmap(df_top10_renewable.set_index('country')[ren_energy_columns], 
                 annot=True, fmt='.0f', cmap='YlOrBr', linewidths=.5, cbar_kws={'label': 'Consumo de energía (TWh)'})

plt.title('Consumo de energías renovables por país (1990 - 2022)')
plt.xlabel('Tipos de energía')
plt.ylabel('País')

plt.tight_layout()

def create_box_plot_plotly(df):
    fig = go.Figure()

    # Crear el gráfico de caja (boxplot)
    fig.add_trace(go.Box(
        y=df['ren_energy'],
        x=df['year'],
        boxpoints='all',  # Mostrar todos los puntos
        marker_color='orange'
    ))

    fig.update_layout(
        title="Distribución del consumo de energías renovables por año (1990 - 2022)",
        xaxis_title="Año",
        yaxis_title="Consumo total de energías renovables (TWh)",
        showlegend=False
    )
    
    return fig

def create_box_plot_after_arima(df):
    fig = go.Figure()

    # Crear el boxplot
    fig.add_trace(go.Box(
        y=df['ren_energy'],
        x=df['year'],
        boxpoints='all',
        marker_color='brown'
    ))

    fig.update_layout(
        title="Consumo de energías renovables por año después de la imputación ARIMA (1990 - 2022)",
        xaxis_title="Año",
        yaxis_title="Consumo total de energías renovables (TWh)",
        showlegend=False
    )
    
    return fig

def create_time_series_plot(df, energy_type):
    fig = go.Figure()

    # Gráfica de la serie original
    fig.add_trace(go.Scatter(x=df['year'], y=df[energy_type], mode='lines+markers',
                             name=f'{energy_type} - Original', line=dict(color='blue')))
    
    # Gráfica de la serie imputada
    fig.add_trace(go.Scatter(x=df['year'], y=df_no_outliers[energy_type], mode='lines',
                             name=f'{energy_type} - Imputado ARIMA', line=dict(color='orange', dash='dash')))
    
    fig.update_layout(
        title=f'Serie de {energy_type}: Original vs Imputado',
        xaxis_title='Año',
        yaxis_title='Consumo (TWh)',
        showlegend=True
    )

    return fig

def generate_time_series_plots(df):
    figs = []
    for energy in ren_energy_columns:
        figs.append(create_time_series_plot(df, energy))
    return figs

def create_histogram(df_original, df_no_outliers, energy_type):
    fig = px.histogram(df_original, x=energy_type, nbins=30, opacity=0.5, 
                       color_discrete_sequence=['blue'], marginal="rug", 
                       labels={'x': 'Consumo (TWh)', 'y': 'Frecuencia'})
    fig.add_histogram(x=df_no_outliers[energy_type], nbinsx=30, opacity=0.5, marker_color='orange', name='Imputado')

    fig.update_layout(
        title=f'Distribución de {energy_type}: Original vs Imputado',
        xaxis_title='Consumo (TWh)',
        yaxis_title='Frecuencia',
        barmode='overlay'
    )
    return fig

def generate_histograms(df_original, df_no_outliers):
    histograms = []
    for energy in ren_energy_columns:
        histograms.append(create_histogram(df_original, df_no_outliers, energy))
    return histograms

def create_kruskal_wallis_table():
    # Formatear los datos para que sean legibles en una tabla
    df_results = pd.DataFrame({
        'Energy Type': ren_energy_columns,
        'Statistic': [186.958, 361.683, 179.566, 185.621, 712.021, 315.607],
        'P-value': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    })
    
    table = dash_table.DataTable(
        data=df_results.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_results.columns],
        style_table={'overflowX': 'auto'},
        style_cell={
            'height': 'auto',
            'minWidth': '50px', 'width': '50px', 'maxWidth': '100px',
            'whiteSpace': 'normal'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'P-value', 'filter_query': '{P-value} = 0'},
                'backgroundColor': '#FF4136',
                'color': 'white'
            }
        ]
    )
    
    return table
# Crear el gráfico de energías renovables
def create_renewable_energy_plot():
    fig = go.Figure()

    # Añadir trazos para cada tipo de energía renovable
    for energy in ren_energy_columns:
        fig.add_trace(go.Scatter(x=df_filtered_plot['year'], 
                                 y=df_filtered_plot[energy], 
                                 mode='lines', 
                                 name=energy))

    # Configurar el título y etiquetas
    fig.update_layout(
        title='Consumo de diferentes fuentes de energía renovables (1990 - 2022)',
        xaxis_title='Año',
        yaxis_title='Consumo (TWh)',
        legend_title='Tipo de Energía',
        template='plotly_white'
    )
    return fig

# Crear el gráfico de energías no renovables
def create_non_renewable_energy_plot():
    fig = go.Figure()

    # Añadir trazos para cada tipo de energía no renovable
    for energy in nonren_energy_columns:
        fig.add_trace(go.Scatter(x=df_filtered_plot_nonren['year'], 
                                 y=df_filtered_plot_nonren[energy], 
                                 mode='lines', 
                                 name=energy))

    # Configurar el título y etiquetas
    fig.update_layout(
        title='Consumo de diferentes fuentes de energía no renovables (1990 - 2022)',
        xaxis_title='Año',
        yaxis_title='Consumo (TWh)',
        legend_title='Tipo de Energía',
        template='plotly_white'
    )
    return fig

# Crear la gráfica para energías renovables
def plot_renewable_bar():
    fig = go.Figure([go.Bar(
        x=df_total_energy_per_year['year'],
        y=df_total_energy_per_year['ren_energy'],
        text=[f"{val/1000:.1f}K" for val in df_total_energy_per_year['ren_energy']],
        textposition='auto',
        marker_color='#90EE90',
    )])

    fig.update_layout(
        title='Suma total de fuentes de energía renovable (1990 - 2022)',
        xaxis_title='Año',
        yaxis_title='Consumo Total (TWh)',
        template='plotly_white'
    )
    
    return fig

# Crear la gráfica para energías no renovables
def plot_non_renewable_bar():
    fig = go.Figure([go.Bar(
        x=df_total_energy_per_year['year'],
        y=df_total_energy_per_year['nonren_energy'],
        text=[f"{val/1000:.1f}K" for val in df_total_energy_per_year['nonren_energy']],
        textposition='auto',
        marker_color='#FFA07A',
    )])

    fig.update_layout(
        title='Suma total de fuentes de energía no renovable (1990 - 2022)',
        xaxis_title='Año',
        yaxis_title='Consumo Total (TWh)',
        template='plotly_white'
    )
    
    return fig

# Gráfico de pie para energías renovables
def plot_renewable_pie():
    fig = go.Figure(go.Pie(
        labels=ren_energy_columns,
        values=df_renewable_total,
        hoverinfo='label+percent',
        textinfo='value',
        marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6'])
    ))
    fig.update_layout(title_text='Porcentaje del consumo total de energías renovables (1990 - 2022)')
    return fig

# Gráfico de pie para energías no renovables
def plot_non_renewable_pie():
    fig = go.Figure(go.Pie(
        labels=nonren_energy_columns,
        values=df_nonrenewable_total,
        hoverinfo='label+percent',
        textinfo='value',
        marker=dict(colors=['#ff6666', '#ffcc66', '#66ff66', '#66ccff'])
    ))
    fig.update_layout(title_text='Porcentaje del consumo total de energías no renovables (1990 - 2022)')
    return fig

# Función para generar el gráfico de coropletas
def plot_map():
    df_renewable_consumption = df_filtered_range.groupby('country')[ren_energy_columns].sum().reset_index()
    df_renewable_consumption['total_renewable'] = df_renewable_consumption[ren_energy_columns].sum(axis=1)

    fig = px.choropleth(df_renewable_consumption,
                        locations="country",
                        locationmode="country names",
                        color="total_renewable",
                        hover_name="country",
                        title="Consumo total de energías renovables por país (1990 - 2022)",
                        color_continuous_scale=px.colors.sequential.YlGnBu,
                        range_color=(0, 150000))
    
    fig.update_layout(geo=dict(
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="white",
        showocean=True,
        oceancolor="LightBlue",
        showcountries=True,
        countrycolor="black"
    ))
    
    return fig

def plot_top10_barchart():
    # Crear gráfico de barras con Plotly
    fig = px.bar(df_top10_renewable, 
                 x='total_renewable', 
                 y='country', 
                 text='total_renewable', 
                 orientation='h', 
                 title='Top 10 países con mayor consumo de energías renovables (1990 - 2022)',
                 labels={'total_renewable':'Consumo Total de Energía (TWh)', 'country':'País'},
                 color_discrete_sequence=['orange'])

    fig.update_traces(texttemplate='%{text:.2f} TWh', textposition='outside')
    fig.update_layout(xaxis_title='Consumo Total de Energía (TWh)', yaxis_title='País')
    
    return fig

def plot_heatmap():
    # Crear el heatmap con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=df_top10_renewable[ren_energy_columns].values,  # Valores del consumo
        x=ren_energy_columns,  # Columnas de energías
        y=df_top10_renewable['country'],  # Países
        colorscale='YlOrBr',  # Escala de color
        colorbar=dict(title='Consumo de energía (TWh)')
    ))
    
    fig.update_layout(
        title='Consumo de energías renovables por país (1990 - 2022)',
        xaxis_title='Tipos de energía',
        yaxis_title='País',
        height=600,
        width=800
    )
    
    return fig

def plot_percentage_consumption():
    # Código existente para generar la gráfica
    fig = px.line(df_annual_consumption, 
                  x='year', 
                  y='percentage_renewable', 
                  title='Porcentaje de consumo de energías renovables vs no renovables (1990 - 2022)',
                  labels={'percentage_renewable': 'Porcentaje de Energías Renovables (%)', 'year': 'Año'},
                  markers=True)
    
    # Ajustar trazado
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='top right')
    
    return fig

# Inicializar la aplicación Dash
app = dash.Dash(__name__)

# Layout de la app
app.layout = html.Div([
    dcc.Tabs(id="tabs-example", value='tab-1', children=[
        dcc.Tab(label='Detección de Valores Atípicos', value='tab-1'),
        dcc.Tab(label='Imputación Arima', value='tab-2'),
        dcc.Tab(label='Distribución de la Imputación - Kruskall Walls', value='tab-3'),
        dcc.Tab(label='Consumo de diferentes fuentes de energía', value='tab-4'),
        dcc.Tab(label='Consumo de energía por países', value='tab-5'),
        dcc.Tab(label='Evolución del porcentaje de energías renovables', value='tab-6'),
    ]),
    html.Div(id='tabs-content-example')
])

# Función de callback para actualizar el contenido basado en la pestaña seleccionada
@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Detección de Valores Atípicos'),
            dcc.Graph(figure=create_box_plot_plotly(df_top10)),
            html.H3('Detección de Valores Atípicos despues de la Imputación ARIMA'),
            dcc.Graph(figure=create_box_plot_after_arima(df_no_outliers))
        ])
    elif tab == 'tab-2':
        figs = generate_time_series_plots(df_filtered_range)  # Generar todas las gráficas
        return html.Div([
            html.H3('Imputación ARIMA'),
            dcc.Graph(figure=figs[0]),  # Primera gráfica (biofuel_consumption)
            dcc.Graph(figure=figs[1]),  # Segunda gráfica (hydro_consumption)
            dcc.Graph(figure=figs[2]),  # Tercera gráfica (solar_consumption)
            dcc.Graph(figure=figs[3]),
            dcc.Graph(figure=figs[4]),
            dcc.Graph(figure=figs[5]),
        ])
    elif tab == 'tab-3':
        histograms = generate_histograms(df_filtered_range, df_no_outliers)
        kruskal_table = create_kruskal_wallis_table()
        return html.Div([
            html.H3('Distribución de la Imputación - Kruskall Walls'),
            dcc.Graph(figure=histograms[0]),  # Primer gráfico de biofuel_consumption
            dcc.Graph(figure=histograms[1]),  # Segundo gráfico de hydro_consumption
            dcc.Graph(figure=histograms[2]),  # Tercer gráfico de solar_consumption
            dcc.Graph(figure=histograms[3]),
            dcc.Graph(figure=histograms[4]), 
            dcc.Graph(figure=histograms[5]),
            html.H3('Distribución de la Imputación - Kruskall Walls'),
            kruskal_table
        ])
    elif tab == 'tab-4':
        renewable_plot = dcc.Graph(figure=create_renewable_energy_plot())
        non_renewable_plot = dcc.Graph(figure=create_non_renewable_energy_plot())
        return html.Div([
            html.H3('Consumo de diferentes fuentes de energía'),
            renewable_plot,
            non_renewable_plot,
            dcc.Graph(figure=plot_renewable_bar()),
            dcc.Graph(figure=plot_non_renewable_bar()),
            dcc.Graph(figure=plot_renewable_pie()),
            dcc.Graph(figure=plot_non_renewable_pie())
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Consumo de energía por países'),
            dcc.Graph(figure=plot_map()),
            dcc.Graph(figure=plot_top10_barchart()),
            dcc.Graph(figure=plot_heatmap())
        ])
    elif tab == 'tab-6':
        return html.Div([
            html.H3('Evolución del porcentaje de energías renovables'),
            dcc.Graph(figure=plot_percentage_consumption())
        ])

# Ejecutar la app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=9000)