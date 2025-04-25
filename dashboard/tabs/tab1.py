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


def calculate_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['bollinger_upper'] = rolling_mean + (rolling_std * num_std)
    df['bollinger_lower'] = rolling_mean - (rolling_std * num_std)
    df['bollinger_mid'] = rolling_mean
    return df


tab1_layout = html.Div(className='tab-content', children=[
    dbc.Row([
        dbc.Col(className='control-panel', children=[
            html.H3("Stock Price Visualization"),
            html.Div(className='input-group', children=[
                html.Label("Select Companies:", className='control-label'),
                dcc.Dropdown(
                    id='company-dropdown',
                    multi=True,
                    placeholder="Select companies to display",
                    className='company-select-dropdown'
                ),
            ]),

            html.Div(className='input-group', children=[
                html.Label("Date Range:", className='control-label'),
                dcc.DatePickerRange(
                    id='date-range-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='vertical',
                    className='date-picker'
                ),
            ]),

            html.Div(className='input-group', children=[
                html.Label("Chart Type:", className='control-label'),
                dcc.RadioItems(
                    id='chart-type',
                    options=[
                        {'label': 'Line Chart', 'value': 'line'},
                        {'label': 'Candlestick', 'value': 'candlestick'}
                    ],
                    value='line',
                    className='radio-items',
                    labelStyle={'display': 'inline-block',
                                'marginRight': '10px'}
                ),
            ]),

            html.Div(className='input-group', children=[
                html.Label("Y-Axis Scale:", className='control-label'),
                dcc.RadioItems(
                    id='y-scale',
                    options=[
                        {'label': 'Linear', 'value': 'linear'},
                        {'label': 'Logarithmic', 'value': 'log'}
                    ],
                    value='linear',
                    className='radio-items',
                    labelStyle={'display': 'inline-block',
                                'marginRight': '10px'}
                ),
            ]),

            html.Div(className='input-group', children=[
                html.Label("Show Bollinger Bands:", className='control-label'),
                dcc.Checklist(
                    id='show-bollinger',
                    options=[
                        {'label': 'Show Bollinger Bands', 'value': 'show'}
                    ],
                    value=[],
                    className='checklist-items',
                ),
            ]),

            html.Div(className='input-group bollinger-settings', children=[
                html.Label("Bollinger Settings:", className='control-label'),
                dbc.Row([
                    dbc.Col([
                        html.Label("Window:", className='sub-label'),
                        dcc.Input(
                            id='bollinger-window',
                            type='number',
                            value=20,
                            min=5,
                            max=100,
                            step=1,
                            className='number-input'
                        ),
                    ], width=6),
                    dbc.Col([
                        html.Label("Std Dev:", className='sub-label'),
                        dcc.Input(
                            id='bollinger-std',
                            type='number',
                            value=2,
                            min=1,
                            max=5,
                            step=0.5,
                            className='number-input'
                        ),
                    ], width=6),
                ]),
            ]),

            html.Div(className='diagnostics-section', children=[
                html.Hr(),
                html.H4("Database Diagnostics"),
                html.Button("Run Database Diagnostics", id="run-diagnostics",
                            n_clicks=0, className='btn btn-secondary'),
                html.Div(id="diagnostics-output",
                         className='diagnostics-output')
            ]),
        ], width=3),

        dbc.Col(className='chart-area', children=[
            dcc.Graph(
                id='stock-price-graph',
                className='main-chart',
                config={'displayModeBar': True}
            ),
        ], width=9),
    ]),
])


@app.callback(
    ddep.Output("company-dropdown", "options"),
    [ddep.Input("company-dropdown", "search_value")]
)
def update_company_options(search_value):
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


@app.callback(
    [ddep.Output("date-range-picker", "min_date_allowed"),
     ddep.Output("date-range-picker", "max_date_allowed"),
     ddep.Output("date-range-picker", "start_date"),
     ddep.Output("date-range-picker", "end_date")],
    [ddep.Input("company-dropdown", "value")]
)
def update_date_picker(selected_companies):
    try:
        min_date, max_date = db.get_date_range()
        end_date = max_date
        start_date = pd.Timestamp(max_date) - pd.Timedelta(days=1200)
        start_date = start_date.strftime('%Y-%m-%d')
        return min_date, max_date, start_date, end_date
    except Exception as e:
        today = datetime.now().strftime('%Y-%m-%d')
        six_months_ago = (
            datetime.now() - pd.Timedelta(days=1200)).strftime('%Y-%m-%d')
        return six_months_ago, today, six_months_ago, today


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
        fig = go.Figure()
        fig.update_layout(
            title="Please select at least one company and date range",
            template="plotly_white",
            xaxis={'visible': False},
            yaxis={'visible': False},
            annotations=[
                dict(
                    text="No data to display",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(
                        size=20
                    )
                )
            ]
        )
        return fig

    fig = go.Figure()

    for company in selected_companies:
        try:
            print(f"DASHBOARD: Getting stock data for {company}")
            stock_data = db.get_company_data(company, start_date, end_date)

            if stock_data.empty:
                print(f"DASHBOARD: No data found for {company}")
                continue

            print(f"DASHBOARD: Received {
                  len(stock_data)} records for {company}")

            if chart_type == 'line':
                fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['close'],
                    mode='lines',
                    name=company
                ))
            else:
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['open'],
                    high=stock_data['high'],
                    low=stock_data['low'],
                    close=stock_data['close'],
                    name=company
                ))

            # and company == selected_companies[0]:
            if 'show' in show_bollinger:
                bb_data = calculate_bollinger_bands(
                    stock_data.copy(), window=bollinger_window, num_std=bollinger_std)

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_upper'],
                    mode='lines',
                    line=dict(width=1, color='rgba(255, 100, 100, 0.6)'),
                    showlegend=False,
                    name='Upper Band'
                ))

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_lower'],
                    mode='lines',
                    line=dict(width=1, color='rgba(100, 100, 255, 0.6)'),
                    fill='tonexty',
                    fillcolor='rgba(150, 150, 150, 0.1)',
                    showlegend=False,
                    name='Lower Band'
                ))

                fig.add_trace(go.Scatter(
                    x=bb_data.index,
                    y=bb_data['bollinger_mid'],
                    mode='lines',
                    line=dict(width=1, dash='dash',
                              color='rgba(100, 100, 100, 0.8)'),
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
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )

    return fig


@app.callback(
    ddep.Output("diagnostics-output", "children"),
    [ddep.Input("run-diagnostics", "n_clicks")]
)
def run_diagnostics(n_clicks):
    if n_clicks == 0:
        return "Click the button to run diagnostics"

    try:
        print("\nDASHBOARD: Running database diagnostics...")

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

        stats = db.get_database_stats()

        diagnostic_results = []

        if "Error" in stats:
            diagnostic_results.append(html.P(
                f"❌ Database error: {stats['Error']}", className='error-message'))
        else:
            diagnostic_results.extend([
                html.P(f"✅ Connected to database: Yes",
                       className='success-message'),
                html.P(
                    f"📊 Total companies: {stats.get('company_count', 'Unknown')}"),
                html.P(
                    f"📈 Companies with stock data: {stats.get('active_companies', 'Unknown')}"),
                html.P(
                    f"🔢 Stock records: {stats.get('stock_count', 'Unknown')}"),
                html.Hr(),
                html.P("See console for complete diagnostic information",
                       className='info-message')
            ])

        return html.Div(diagnostic_results)
    except Exception as e:
        error_msg = str(e)
        print(f"DASHBOARD ERROR in diagnostics: {error_msg}")
        return html.P(f"❌ Error running diagnostics: {error_msg}", className='error-message')
