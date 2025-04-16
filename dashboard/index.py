import pandas as pd

from dash import html
import dash.dependencies as ddep
from dash import dcc
import dash_bootstrap_components as dbc
#import dash_extensions as de

from tabs.tab1 import tab1_layout
from tabs.tab2 import tab2_layout
from tabs.tab3 import tab3_layout

from app import app, db

# Création de la présentation Dash
layout = dbc.Container([
    html.H1("Un joli titre"),
    dbc.Tabs(
        id="tabs-example",
        active_tab="tab-1",
        children = [
            dbc.Tab(label="Graphique", tab_id="tab-1"),
            dbc.Tab(label="Tableau", tab_id="tab-2"),
            dbc.Tab(label="SQL", tab_id="tab-3"),
        ],
    ),
    html.Div(id="tabs-content"),
])

# Callback to update the dropdown options based on input text
@app.callback(
    ddep.Output("tabs-content", "children"),
    [ddep.Input("tabs-example", "active_tab")],
)
def render_content(tab):
    if tab == "tab-1":
        return tab1_layout
    elif tab == "tab-2":
        return tab2_layout
    elif tab == "tab-3":
        return tab3_layout

