import pandas as pd

from dash import dcc
from dash import html
from dash import dash_table
import dash.dependencies as ddep
import dash_bootstrap_components as dbc

from app import app, db

tab2_layout = html.Div(className='tab-content', children=[
    dbc.Row([
        dbc.Col(className='control-panel', children=[
            html.H3("Stock Data Table"),
            html.Div(className='input-group', children=[
                html.Label("Select Companies:", className='control-label'),
                dcc.Dropdown(
                    id='table-company-dropdown',
                    multi=True,
                    placeholder="Select companies to display",
                    className='company-select-dropdown'  # Added className
                ),
            ]),

            html.Div(className='input-group', children=[
                html.Label("Date Range:", className='control-label'),
                dcc.DatePickerRange(
                    id='table-date-range-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='vertical',
                    className='date-picker'
                ),
            ]),

            html.Button('Generate Table', id='generate-table-button',
                        className='btn btn-primary generate-button'),
        ], width=3),

        dbc.Col(className='table-area', children=[
            html.Div(id="stats-container", className="table-container"),
        ], width=9),
    ]),
])


@app.callback(
    ddep.Output("table-company-dropdown", "options"),
    [ddep.Input("table-company-dropdown", "search_value")]
)
def update_table_company_options(search_value):
    try:
        companies = db.get_companies()
        options = [{'label': company, 'value': company}
                   for company in companies]
        return options
    except Exception as e:
        print(f"DASHBOARD ERROR in table company dropdown: {e}")
        return []


@app.callback(
    [ddep.Output("table-date-range-picker", "min_date_allowed"),
     ddep.Output("table-date-range-picker", "max_date_allowed"),
     ddep.Output("table-date-range-picker", "start_date"),
     ddep.Output("table-date-range-picker", "end_date")],
    [ddep.Input("table-company-dropdown", "value")]
)
def update_table_date_picker(selected_companies):
    try:
        min_date, max_date = db.get_date_range()
        end_date = max_date
        start_date = pd.Timestamp(max_date) - pd.Timedelta(days=30)
        start_date = start_date.strftime('%Y-%m-%d')
        return min_date, max_date, start_date, end_date
    except Exception as e:
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        thirty_days_ago = (pd.Timestamp.now() -
                           pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        return thirty_days_ago, today, thirty_days_ago, today


@app.callback(
    ddep.Output("stats-container", "children"),
    [ddep.Input("generate-table-button", "n_clicks")],
    [ddep.State("table-company-dropdown", "value"),
     ddep.State("table-date-range-picker", "start_date"),
     ddep.State("table-date-range-picker", "end_date")]
)
def generate_stats_table(n_clicks, selected_companies, start_date, end_date):
    if not n_clicks or not selected_companies or not start_date or not end_date:
        return html.Div("Select companies and date range, then click 'Generate Table'", className='placeholder-message')

    try:
        all_stats = []

        for company in selected_companies:
            stock_data = db.get_company_data(company, start_date, end_date)

            if stock_data.empty:
                continue

            daily_data = stock_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            if daily_data.empty:
                continue

            stats_df = pd.DataFrame({
                'Date': daily_data.index.strftime('%Y-%m-%d'),
                'Company': company,
                'Open': daily_data['open'].round(2),
                'High': daily_data['high'].round(2),
                'Low': daily_data['low'].round(2),
                'Close': daily_data['close'].round(2),
                'Min': daily_data['low'].round(2),
                'Max': daily_data['high'].round(2),
                'Mean': daily_data[['open', 'high', 'low', 'close']].mean(axis=1).round(2),
                'Std': daily_data[['open', 'high', 'low', 'close']].std(axis=1).round(2),
                'Volume': daily_data['volume']
            })

            all_stats.append(stats_df)

        if not all_stats:
            return html.Div("No data available for the selected companies and date range", className='info-message')

        combined_stats = pd.concat(all_stats)
        table_data = combined_stats.to_dict('records')

        return [
            html.H4("Daily Stock Statistics"),
            dash_table.DataTable(
                id='stats-table',
                columns=[
                    {'name': 'Date', 'id': 'Date', 'type': 'datetime'},
                    {'name': 'Company', 'id': 'Company'},
                    {'name': 'Open', 'id': 'Open', 'type': 'numeric'},
                    {'name': 'High', 'id': 'High', 'type': 'numeric'},
                    {'name': 'Low', 'id': 'Low', 'type': 'numeric'},
                    {'name': 'Close', 'id': 'Close', 'type': 'numeric'},
                    {'name': 'Min', 'id': 'Min', 'type': 'numeric'},
                    {'name': 'Max', 'id': 'Max', 'type': 'numeric'},
                    {'name': 'Mean', 'id': 'Mean', 'type': 'numeric'},
                    {'name': 'Std', 'id': 'Std', 'type': 'numeric'},
                    {'name': 'Volume', 'id': 'Volume',
                        'type': 'numeric', 'format': {'specifier': ','}},
                ],
                data=table_data,
                page_size=20,
                sort_action='native',
                filter_action='native',
                fixed_rows={'headers': True},
                style_table={'height': '65vh',
                             'overflowY': 'auto', 'overflowX': 'auto'},
                style_cell={
                    'minWidth': '90px', 'width': '110px', 'maxWidth': '150px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'textAlign': 'left',
                    'padding': '8px'
                },
                style_header={
                    'backgroundColor': '#f8f9fa',
                    'fontWeight': 'bold',
                    'borderBottom': '1px solid #dee2e6'
                },
                style_data={
                    'borderBottom': '1px solid #dee2e6',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#fdfdfe'
                    }
                ],
            )
        ]

    except Exception as e:
        return html.Div(f"Error generating table: {str(e)}", className='error-message')
