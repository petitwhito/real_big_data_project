import pandas as pd

from dash import dcc
from dash import html
from dash import dash_table
import dash.dependencies as ddep
import dash_bootstrap_components as dbc

from app import app, db

tab3_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H3("SQL Query Interface"),
            dcc.Textarea(
                id='sql-query-input',
                placeholder="Enter your SQL query here...",
                style={'width': '100%', 'height': 200},
            ),
            html.Button('Execute Query', id='execute-query-button',
                        className='btn btn-primary my-3'),
            html.Div(id='sql-query-error', style={'color': 'red'}),
        ], width=12),

        dbc.Col([
            html.H4("Query Result"),
            html.Div(id='sql-query-result-container'),
        ], width=12),
    ]),
])

# Callback to execute SQL query


@app.callback(
    [ddep.Output("sql-query-result-container", "children"),
     ddep.Output("sql-query-error", "children")],
    [ddep.Input("execute-query-button", "n_clicks")],
    [ddep.State("sql-query-input", "value")]
)
def execute_sql_query(n_clicks, query):
    if not n_clicks or not query:
        return html.Div("Enter a SQL query and click 'Execute Query'"), ""

    try:
        # Execute the query through the database model
        results = db.execute_query(query)

        if isinstance(results, pd.DataFrame):
            if results.empty:
                return html.Div("Query executed successfully, but returned no data."), ""

            # Convert to dictionary for dash_table
            table_data = results.to_dict('records')
            columns = [{"name": str(col), "id": str(col)}
                       for col in results.columns]

            # Create the table
            result_table = dash_table.DataTable(
                id='sql-result-table',
                columns=columns,
                data=table_data,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'minWidth': '100px',
                    'maxWidth': '300px',
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

            return result_table, ""
        else:
            return html.Div(f"Query executed successfully. Rows affected: {results}"), ""

    except Exception as e:
        return html.Div(), f"Error executing query: {str(e)}"
