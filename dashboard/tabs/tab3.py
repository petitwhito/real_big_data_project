import pandas as pd

from dash import dcc
from dash import html
from dash import dash_table
import dash.dependencies as ddep
import dash_bootstrap_components as dbc

from app import app, db

tab3_layout = html.Div(className='tab-content', children=[
    dbc.Row([
        dbc.Col(className='sql-input-section', children=[
            html.H3("SQL Query Interface"),
            dcc.Textarea(
                id='sql-query-input',
                placeholder="Enter your SQL query here (e.g., SELECT DISTINCT company FROM companies LIMIT 10;)",
                className='sql-textarea',
            ),
            html.Button('Execute Query', id='execute-query-button',
                        className='btn btn-primary execute-button'),
            html.Div(id='sql-query-error', className='error-message'),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col(className='sql-output-section', children=[
            html.H4("Query Result"),
            html.Div(id='sql-query-result-container',
                     className='table-container'),
        ], width=12),
    ]),
])


@app.callback(
    [ddep.Output("sql-query-result-container", "children"),
     ddep.Output("sql-query-error", "children")],
    [ddep.Input("execute-query-button", "n_clicks")],
    [ddep.State("sql-query-input", "value")]
)
def execute_sql_query(n_clicks, query):
    if not n_clicks or not query:
        return html.Div("Enter a SQL query and click 'Execute Query'", className='placeholder-message'), ""

    error_output = ""
    result_output = ""

    try:
        results = db.execute_query(query)

        if isinstance(results, pd.DataFrame):
            if results.empty:
                result_output = html.Div(
                    "Query executed successfully, but returned no data.", className='info-message')
            else:
                table_data = results.to_dict('records')
                columns = [{"name": str(col), "id": str(col)}
                           for col in results.columns]

                result_table = dash_table.DataTable(
                    id='sql-result-table',
                    columns=columns,
                    data=table_data,
                    page_size=20,
                    sort_action='native',
                    filter_action='native',
                    fixed_rows={'headers': True},
                    style_table={'height': '60vh',
                                 'overflowY': 'auto', 'overflowX': 'auto'},
                    style_cell={
                        'minWidth': '100px', 'width': '150px', 'maxWidth': '300px',
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
                result_output = result_table
        else:
            result_output = html.Div(f"Query executed successfully. Rows affected: {
                                     results}", className='success-message')

    except Exception as e:
        error_output = f"Error executing query: {str(e)}"

    return result_output, error_output
