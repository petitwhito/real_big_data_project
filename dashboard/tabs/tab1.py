import pandas as pd
import numpy as np

from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash.dependencies as ddep
import dash_bootstrap_components as dbc
import plotly.express as px
from datetime import datetime

from app import app, db

# Function to calculate Bollinger Bands


def calculate_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['bollinger_upper'] = rolling_mean + (rolling_std * num_std)
    df['bollinger_lower'] = rolling_mean - (rolling_std * num_std)
    df['bollinger_mid'] = rolling_mean
    return df


tab1_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("Stock Price Visualization"),
            html.Div([
                html.Label("Select Companies:"),
                dcc.Dropdown(
                    id='company-dropdown',
                    multi=True,
                    placeholder="Select companies to display"
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='vertical',
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Chart Type:"),
                dcc.RadioItems(
                    id='chart-type',
                    options=[
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Candlestick', 'value': 'candlestick'}
                    ],
                    value='line',
                    labelStyle={'display': 'inline-block',
                                'marginRight': '10px'}
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Y-Axis Scale:"),
                dcc.RadioItems(
                    id='y-scale',
                    options=[
                        {'label': 'Linear', 'value': 'linear'},
                        {'label': 'Logarithmic', 'value': 'log'}
                    ],
                    value='linear',
                    labelStyle={'display': 'inline-block',
                                'marginRight': '10px'}
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Show Bollinger Bands:"),
                dcc.Checklist(
                    id='show-bollinger',
                    options=[
                        {'label': 'Show Bollinger Bands', 'value': 'show'}
                    ],
                    value=[],
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Bollinger Settings:"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Window:"),
                        dcc.Input(
                            id='bollinger-window',
                            type='number',
                            value=20,
                            min=5,
                            max=100,
                            step=1
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Standard Deviations:"),
                        dcc.Input(
                            id='bollinger-std',
                            type='number',
                            value=2,
                            min=1,
                            max=5,
                            step=0.5
                        ),
                    ], width=6),
                ]),
            ], style={'marginBottom': '20px'}),
            
            # Added diagnostics section
            html.Div([
                html.Hr(),
                html.H4("Database Diagnostics"),
                html.Button("Run Database Diagnostics", id="run-diagnostics", n_clicks=0),
                html.Div(id="diagnostics-output")
            ], style={'marginTop': '20px'}),
        ], width=3),

        dbc.Col([
            dcc.Graph(
                id='stock-price-graph',
                style={'height': '80vh'},
                config={'displayModeBar': True}
            ),
        ], width=9),
    ]),
])

# Callback to update company dropdown options


@app.callback(
    ddep.Output("company-dropdown", "options"),
    [ddep.Input("company-dropdown", "search_value")]
)
def update_company_options(search_value):
    # Get company names from the database
    print("DASHBOARD: Updating company dropdown options...")
    try:
        companies = db.get_companies()
        options = [{'label': company, 'value': company}
                   for company in companies]
        print(f"DASHBOARD: Dropdown loaded with {len(options)} companies")
        return options
    except Exception as e:
        print(f"DASHBOARD ERROR in company dropdown: {e}")
        return []

# Callback to initialize date picker


@app.callback(
    [ddep.Output("date-range-picker", "min_date_allowed"),
     ddep.Output("date-range-picker", "max_date_allowed"),
     ddep.Output("date-range-picker", "start_date"),
     ddep.Output("date-range-picker", "end_date")],
    [ddep.Input("company-dropdown", "value")]
)
def update_date_picker(selected_companies):
    # Set default date range
    try:
        min_date, max_date = db.get_date_range()
        # Set default date range to the last 6 months
        end_date = max_date
        start_date = pd.Timestamp(max_date) - pd.Timedelta(days=1200)
        start_date = start_date.strftime('%Y-%m-%d')
        return min_date, max_date, start_date, end_date
    except Exception as e:
        # Default fallback values
        today = datetime.now().strftime('%Y-%m-%d')
        six_months_ago = (
            datetime.now() - pd.Timedelta(days=1200)).strftime('%Y-%m-%d')
        return six_months_ago, today, six_months_ago, today

# Callback to update the stock price graph


@app.callback(
    ddep.Output("stock-price-graph", "figure"),
    [ddep.Input("company-dropdown", "value"),
     ddep.Input("date-range-picker", "start_date"),
     ddep.Input("date-range-picker", "end_date"),
     ddep.Input("chart-type", "value"),
     ddep.Input("y-scale", "value"),
     ddep.Input("show-bollinger", "value"),
     ddep.Input("bollinger-window", "value"),
     ddep.Input("bollinger-std", "value")]
)
def update_stock_graph(selected_companies, start_date, end_date, chart_type, y_scale, show_bollinger, bollinger_window, bollinger_std):
    if not selected_companies or not start_date or not end_date:
        return go.Figure().update_layout(title="Please select at least one company and date range")

    fig = go.Figure()

    for company in selected_companies:
        try:
            # Get stock data from database
            print(f"DASHBOARD: Getting stock data for {company}")
            stock_data = db.get_company_data(company, start_date, end_date)

            if stock_data.empty:
                print(f"DASHBOARD: No data found for {company}")
                continue

            print(f"DASHBOARD: Received {len(stock_data)} records for {company}")
            
            if chart_type == 'line':
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['close'],
                    mode='lines',
                    name=company
                ))
            else:  # candlestick
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['open'],
                    high=stock_data['high'],
                    low=stock_data['low'],
                    close=stock_data['close'],
                    name=company
                ))

            # Add Bollinger Bands if requested (only for the first company)
            if 'show' in show_bollinger and company == selected_companies[0]:
                bb_data = calculate_bollinger_bands(
                    stock_data.copy(), window=bollinger_window, num_std=bollinger_std)

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_upper'],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(200, 0, 0, 0.5)'),
                    name='Upper Band'
                ))

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_lower'],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(200, 0, 0, 0.5)'),
                    fill='tonexty',
                    fillcolor='rgba(200, 0, 0, 0.1)',
                    name='Lower Band'
                ))

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_mid'],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(200, 0, 0, 0.9)'),
                    name='SMA'
                ))

        except Exception as e:
            print(f"DASHBOARD ERROR rendering chart for {company}: {str(e)}")
            continue

    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis_type=y_scale,
        legend_title="Companies",
        template="plotly_white",
        height=700
    )

    return fig

# Added diagnostic callback
@app.callback(
    ddep.Output("diagnostics-output", "children"),
    [ddep.Input("run-diagnostics", "n_clicks")]
)
def run_diagnostics(n_clicks):
    if n_clicks == 0:
        return "Click the button to run diagnostics"
    
    try:
        print("\nDASHBOARD: Running database diagnostics...")
        
        # First, try a direct connection test
        try:
            print("Testing direct connection with psycopg2...")
            import psycopg2
            conn = psycopg2.connect(
                database="bourse",
                user="ricou",
                host="db",
                password="monmdp"
            )
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            print("Direct connection successful")
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Direct connection test failed: {str(e)}")
        
        # Now run our standard diagnostics
        stats = db.get_database_stats()
        
        diagnostic_results = []
        
        if "Error" in stats:
            diagnostic_results.append(html.P(f"❌ Database error: {stats['Error']}", style={'color': 'red'}))
        else:
            diagnostic_results.extend([
                html.P(f"✅ Connected to database: Yes", style={'color': 'green'}),
                html.P(f"📊 Total companies: {stats.get('company_count', 'Unknown')}"),
                html.P(f"📈 Companies with stock data: {stats.get('active_companies', 'Unknown')}"),
                html.P(f"🔢 Stock records: {stats.get('stock_count', 'Unknown')}"),
                html.Hr(),
                html.P("See console for complete diagnostic information")
            ])
        
        return html.Div(diagnostic_results)
    except Exception as e:
        error_msg = str(e)
        print(f"DASHBOARD ERROR in diagnostics: {error_msg}")
        return html.P(f"❌ Error running diagnostics: {error_msg}", style={'color': 'red'})