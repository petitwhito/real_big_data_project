import pandas as pd

from dash import dcc
from dash import html
from dash import dash_table
import dash.dependencies as ddep
import dash_bootstrap_components as dbc

from app import app, db

tab2_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("Stock Data Table"),
            html.Div([
                html.Label("Select Companies:"),
                dcc.Dropdown(
                    id='table-company-dropdown',
                    multi=True,
                    placeholder="Select companies to display"
                ),
            ], style={'marginBottom': '20px'}),

            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='table-date-range-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='vertical',
                ),
            ], style={'marginBottom': '20px'}),

            html.Button('Generate Table', id='generate-table-button',
                        className='btn btn-primary mb-3'),
        ], width=3),

        dbc.Col([
            html.Div(id="stats-container", className="mt-4"),
        ], width=9),
    ]),
])

# Callback to update company dropdown options in the table tab


@app.callback(
    ddep.Output("table-company-dropdown", "options"),
    [ddep.Input("table-company-dropdown", "search_value")]
)
def update_table_company_options(search_value):
    # Get company names from the database
    try:
        companies = db.get_companies()
        options = [{'label': company, 'value': company}
                   for company in companies]
        return options
    except Exception as e:
        return []

# Callback to initialize date picker in the table tab


@app.callback(
    [ddep.Output("table-date-range-picker", "min_date_allowed"),
     ddep.Output("table-date-range-picker", "max_date_allowed"),
     ddep.Output("table-date-range-picker", "start_date"),
     ddep.Output("table-date-range-picker", "end_date")],
    [ddep.Input("table-company-dropdown", "value")]
)
def update_table_date_picker(selected_companies):
    # Set default date range
    try:
        min_date, max_date = db.get_date_range()
        # Set default date range to the last 30 days
        end_date = max_date
        start_date = pd.Timestamp(max_date) - pd.Timedelta(days=30)
        start_date = start_date.strftime('%Y-%m-%d')
        return min_date, max_date, start_date, end_date
    except Exception as e:
        # Default fallback values
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        thirty_days_ago = (pd.Timestamp.now() -
                           pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        return thirty_days_ago, today, thirty_days_ago, today

# Callback to generate the table with statistics


@app.callback(
    ddep.Output("stats-container", "children"),
    [ddep.Input("generate-table-button", "n_clicks")],
    [ddep.State("table-company-dropdown", "value"),
     ddep.State("table-date-range-picker", "start_date"),
     ddep.State("table-date-range-picker", "end_date")]
)
def generate_stats_table(n_clicks, selected_companies, start_date, end_date):
    if not n_clicks or not selected_companies or not start_date or not end_date:
        return html.Div("Select companies and date range, then click 'Generate Table'")

    try:
        # Create a list to hold all company data frames
        all_stats = []

        for company in selected_companies:
            # Get stock data from database
            stock_data = db.get_company_data(company, start_date, end_date)

            if stock_data.empty:
                continue

            # Resample to daily data if it's not already daily
            daily_data = stock_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Calculate daily statistics
            stats_df = pd.DataFrame({
                'Date': daily_data.index,
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
            return html.Div("No data available for the selected companies and date range")

        # Combine all company data
        combined_stats = pd.concat(all_stats)

        # Convert to dictionary for dash_table
        table_data = combined_stats.to_dict('records')

        # Create the table
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
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '80px', 'width': '100px', 'maxWidth': '120px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                },
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                page_size=15,
                sort_action='native',
                filter_action='native',
            )
        ]

    except Exception as e:
        return html.Div(f"Error generating table: {str(e)}")
