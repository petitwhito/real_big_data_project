import pandas as pd

from dash import html
import dash.dependencies as ddep
from dash import dcc
import dash_bootstrap_components as dbc
# import dash_extensions as de

from tabs.tab1 import tab1_layout
from tabs.tab2 import tab2_layout
from tabs.tab3 import tab3_layout

from app import app, db

# Creation of the main layout of the app
layout = dbc.Container(fluid=True, className='app-container', children=[
    html.H1("Magnificent Stock WebApp", className='app-title'),
    dbc.Tabs(
        id="tabs-example",
        active_tab="tab-1",
        className='app-tabs',
        children=[
            dbc.Tab(label="Visualization", tab_id="tab-1",
                    className='custom-tab'),
            dbc.Tab(label="Data Table", tab_id="tab-2",
                    className='custom-tab'),
            dbc.Tab(label="SQL Query", tab_id="tab-3", className='custom-tab'),
        ],
    ),
    html.Div(id="tabs-content", className='tab-content-container'),
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
    return html.P("This shouldn't be displayed")
