
import dash
from dash import html
from dash.dependencies import Input, Output
from dash import dcc
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import pmdarima as pm
import plotly.graph_objects as go
# import dash_core_components as dcc
from flask_caching import Cache
from pages.DescriptiveStatistics import DescriptiveStatistics
from pages.CrossCorrelation import CrossCorrelation
from pages.AutoCorrelation import AutoCorrelation
# from pages.SentimentAnalysis import avg_sentiment_scores
from dash_dangerously_set_inner_html import DangerouslySetInnerHTML
from pages.ARIMA import ARIMA
from pages.BERT import BERT
from pages.AboutResearchers import AboutResearchers
from pages.ContactUs import ContactUs
from pages.Projectdetails import Projectdetails
from pages.SentimentAnalysis import layout 
import dash_bootstrap_components as dbc
import pickle
import os
from dash.dependencies import Input, Output

# ,external_stylesheets=[dbc.themes.CERULEAN]

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = '5g and corona-analyze twitter'

# Initialize the cache
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})
cache.clear()
########################      AutoCorrelation  ###############

pivot_df = pd.read_csv('pivot_df.csv')
tweets_per_day_by_lang = pd.read_csv('tweets_per_day_by_lang.csv')

languages = tweets_per_day_by_lang['lang'].unique()
#################################################################
app.layout = html.Div([
    html.H1('5G Conspiracy Theory in Corona', style={'text-align': 'center','color': 'white'}),
    html.P('By Mengsha  Ataly and Yakir Attias', style={'text-align': 'center','color': 'white'}),
    dcc.Location(id='url', refresh=False),
    dcc.Tabs(id="tabs", value='/', children=[
        dcc.Tab(label='Home', value='/'),
        dcc.Tab(label='About the Researchers', value='/AboutResearchers'),
        dcc.Tab(label='Project Details', value='/project-details'),
        dcc.Tab(label='Contact Us', value='/contact-us'),
    ]),
    html.Div(id='tabs-content', className='site-container')
], className='site-container')

@app.callback(Output('tabs-content', 'children'),
              [Input('url', 'pathname')])
def render_content(pathname):
    if pathname == '/':
        return html.Div([
            html.H3('Short Explanation',style={'text-align': 'center','color': 'white'}),
            html.P('The 5G conspiracy theory in relation to the Corona pandemic suggests a false claim that 5G technology is somehow linked to the spread of the virus. This theory has gained traction in certain online communities despite being debunked by scientific evidence and experts.',style={'text-align': 'left','color': 'white'}),
            html.Div(id='tabs-and-content', children=[
                dcc.Tabs(id='nested-tabs', value='nested-tab-1',vertical=True, children=[
                    dcc.Tab(label='Descriptive Statistics', value='nested-tab-1'),
                    dcc.Tab(label='Cross Correlation', value='nested-tab-2'),
                    dcc.Tab(label='Auto Correlation', value='nested-tab-3'),
                    dcc.Tab(label='Sentiment Analysis', value='nested-tab-4'),
                    dcc.Tab(label='Auto ARIMA', value='nested-tab-5'),
                    dcc.Tab(label='BERT', value='nested-tab-6'),
                ]),
            html.Div(id='nested-tabs-content')
            ])
        ])
    elif pathname == '/AboutResearchers':
        return AboutResearchers()
    elif pathname == '/project-details':
        return Projectdetails()
    elif pathname == '/contact-us':
        return ContactUs()

@app.callback(Output('nested-tabs-content', 'children'),
              [Input('nested-tabs', 'value')])
@cache.memoize()  # Use the cache to store the results of this function
def render_nested_content(tab_value):
    if tab_value == 'nested-tab-1':
        return DescriptiveStatistics()
    elif tab_value == 'nested-tab-2':
        return CrossCorrelation()
    elif tab_value == 'nested-tab-3':
        return AutoCorrelation(pivot_df)
    elif tab_value == 'nested-tab-4':
        return layout
    elif tab_value == 'nested-tab-5':
        return ARIMA(languages)
    elif tab_value == 'nested-tab-6':
        return BERT()

@app.callback(Output('url', 'pathname'),
              [Input('tabs', 'value')])
def update_url(tab_value):
    return tab_value

######################################### autocorrelation ################################
@app.callback(
    Output('autocorrelation-graph', 'figure'),
    Input('language-dropdown', 'value')
)
def update_graph(selected_language):
    lags = [1, 2, 3, 4, 5]
    autocorrelation_values = [pivot_df[selected_language].autocorr(lag=lag) for lag in lags]

    figure = go.Figure(data=go.Scatter(x=lags, y=autocorrelation_values))
    figure.update_layout(
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        autosize=False,
        width=950,
        height=300
    )

    return figure

#################################### ARIMA #################################

@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('forecast-values', 'children')],
    [Input('language-dropdown', 'value')]
)
def update_plot(selected_language):
    # Load precomputed data for the selected language
    data_path = f'autoarima/{selected_language}.pkl'
    with open(data_path, 'rb') as file:
        train, test, future_forecast, forecast_index, rmse = pickle.load(file)

    # Create the plot using Plotly
    fig = {
        'data': [
            {'x': train.index, 'y': train, 'type': 'line', 'name': 'Train'},
            {'x': test.index, 'y': test, 'type': 'line', 'name': 'Test'},
            {'x': forecast_index, 'y': future_forecast, 'type': 'line', 'name': 'Forecast'}
        ]
    }

    # Format the forecast values for display
    forecast_values_text = f"Forecast for the next {len(forecast_index)} days for language {selected_language}: {', '.join(map(str, future_forecast))}"

    return fig, forecast_values_text

############################################### sentiment ############################


daily_sentiment_avg = pd.read_csv('daily_sentiment_avg.csv')
daily_sentiment_avg['date'] = pd.to_datetime(daily_sentiment_avg['date'])

@app.callback(
    Output('sentiment-line-chart', 'figure'),
    Input('language-dropdown', 'value')
)
def update_line_chart(selected_lang):
    if selected_lang == 'all':
        # Use data for all languages
        filtered_df = daily_sentiment_avg
    else:
        # Filter data for the selected language
        filtered_df = daily_sentiment_avg[daily_sentiment_avg['lang'] == selected_lang]

    # Create a line chart
    fig = px.line(filtered_df, x='date', y='average_sentiment_polarity', color='lang',
                  title=f'Daily Average Sentiment Polarity for {selected_lang}',
                  labels={'average_sentiment_polarity': 'Sentiment Polarity'})

    return fig


########################################################################## name

AutoCorrelation(pivot_df)
if __name__ == '__main__':
    app.run_server(debug=True)
    
