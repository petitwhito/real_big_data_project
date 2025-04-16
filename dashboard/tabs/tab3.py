from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash.dependencies as ddep
import dash_extensions as de

from app import app, db  # Import these instances

tab3_layout = dcc.Tab(label='SQL', children=[
    html.H2("SQL Terminal"),
    html.Div(id='sql-query-output', style={'whiteSpace': 'pre-line', 'overflowY': 'auto', 'height': 500, 
                                           'border': '1px solid #ccc', 'padding': '10px'}),
    de.Keyboard(
        dcc.Textarea(
            id='sql-query-input',
            style={'width': '100%', 'height': "2em"},
            placeholder='Enter your SQL query here...',
        ),
        captureKeys=["Enter"],
        id='sql-query-key'
    )
])

@app.callback(
    [ddep.Output('sql-query-output', 'children'),
    ddep.Output('sql-query-input', 'value')],
    ddep.Input('sql-query-key', 'n_keydowns'),
    ddep.State('sql-query-input', 'value'),
    ddep.State('sql-query-output', 'children')
)
def execute_query(n_key, query, history):
    if history is None:
        history = []
    if n_key is None or not query:
        return history, query
    try:
        result_df = db.df_query(query)
        if len(result_df) == 0:
            error_msg = str(db.logger.get_last_message())
            try:
                error_msg = error_msg.split(") ")[1].split("\n\n")[0] # used for psycopg2 error messages
            except:
                pass
            if error_msg != "df_query: " + query:  # I use the fact that df_query logs all commands
                return history + [ html.Span("error ", style={'color': 'red'}), html.Pre(error_msg), html.Br() ], query
        return history + [html.Pre([html.B(query), result_df.to_string()]), html.Br()], ""
    except Exception as e:
        return history + [html.Pre(str(e)),], query
