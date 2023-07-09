#importing required libraries and dependencies
import base64
import dash
import dash_bootstrap_components as dbc
import io
import pickle
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import dash_daq as daq
import lasio
# Modelling
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option
set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_predict


# For Bootstrap Icons and Themes
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP]
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem 1rem 1rem",
    "background-color": "#e6e6e6",
}


# the styles for the main content position it to the right of the sidebar
CONTENT_STYLE = {
    "margin-left": "18rem",
    #"margin-right": "1rem",
    #"padding": "1rem 1rem",
    
}

#Define sidebar layout

sidebar = html.Div(
    [
        html.H3("Lithoscope", className="display-4", style={"font-size": "3rem", "font-weight": "bold", "color": "#007bff"}),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink([html.I(className="bi bi-house-fill me-2"), " Home"], href="/", active="exact",),
                dbc.NavLink([html.I(className="bi bi-arrow-bar-up me-2"), " Import data"], href="/page-1", active="exact",),
                dbc.NavLink([html.I(className="bi bi-bar-chart-fill me-2"), " Visualize data"], href="/page-2", active="exact",),
                dbc.NavLink([html.I(className="bi bi-cpu-fill me-2"), " Fitting model"], href="/page-3", active="exact",),
                dbc.NavLink([html.I(className="bi bi-clipboard-data-fill me-2"), " Predict facies"], href="/page-4", active="exact",),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


content = html.Div(id="page-content", style=CONTENT_STYLE)

#Define App Layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])




#home layout
# Define the background image URL and center text
text = 'Lithofacies Identification'
background_image ="https://images.pexels.com/photos/162568/oil-pump-jack-sunset-clouds-silhouette-162568.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"

# Define the layout of the app with the background image, centered text, and a paragraph
home_layout = html.Div(
    style={
        'background-image': f'url("{background_image}")',  # Set the background image URL
        'background-size': 'cover',  # Make sure the image covers the entire background
        'background-position': 'center',  # Center the background image
        'height': '100vh',  # Set the height of the container to the full height of the viewport
        'display': 'flex',  # Use a flexbox layout
        'flex-direction': 'column',  # Stack the elements vertically
        'align-items': 'center',  # Center the child elements horizontally
        'justify-content': 'center',  # Center the child elements vertically
        'font-family': 'Helvetica, Arial, sans-serif'  # Set the font family for the text
    },
    children=[
        html.H1(
            children=text,  # Use the text variable for the header text
            style={
                'font-size': '4.5em',  # Set the font size for the header
                "font-weight": "bold",  # Make the text bold
                'color': 'white',  # Set the text color to white
                'text-shadow': '2px 2px 4px #000000'  # Add a subtle text shadow for contrast
            }
        ),
        html.P(
            'Uncover the Hidden Layers of Earth with Intuitive Lithofacies Detection and Visualization',  # Add a description for the app
            style={
                'font-size': '1.5em',  # Set the font size for the description
                'color': 'white',  # Set the text color to white
                'text-align': 'center',  # Center the text horizontally
                'max-width': '800px',  # Set a maximum width for the text container
                'margin-top': '30px',  # Add some space above the text
                'text-shadow': '1px 1px 2px #000000'  # Add a subtle text shadow for contrast
            }
        )
    ]
)



#page_1_layout

#ref : https://hellodash.pythonanywhere.com/
# Define the header with a title
header = html.H2(
    "Import the Dataset", className="bg-primary text-white p-2 mb-2 text-center"
)

# Define the layout for the first page using a dbc.Container component
page_1_layout = dbc.Container(children=[
    html.Div([ 
        # Include the header and a dcc.Upload component for file uploads
        header,
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'font-size':'25px',
                'width': '100%',
                'height': '55px',
                'lineHeight': '55px',
                'borderWidth': '2px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=True
        ),
        
        # Add a button to add a well, and a span to display the number of wells added
        html.Br(),
        dbc.Button("Add Well",id="add_well", outline=True, color="primary", className="me-1",n_clicks=0),
        html.Span(id="example-output", style={"verticalAlign": "middle"}),
        html.Br(),
        html.Br(),
        
        # Display the data table and graph side-by-side using dbc.Row and dbc.Col components
        dbc.Row(
            [
                dbc.Col(id='output-datatable', md=12),
                dbc.Col(dcc.Graph(id="well_plot"), md=12),
            ],
            align="center",
        ),
        
    ])
])



#page_2_layout____INPUT VISUALIZATION

df = pd.read_csv("assets/facies_vectors.csv")
df.head()
x=df['Well Name'].unique()
x=list(x)
x.append("ALL WELLS")
y=df.columns
dfx=df
y1=dfx.columns[1:7]

header1 = html.H2(
    "Input Data visualization", className="bg-primary text-white p-2 mb-2 text-center"
)

# Create controls for selecting well and features for a crossplot
controls = dbc.Card(
    [
        
         html.Div(
            [
                dbc.Label("Select well"),
                dcc.Dropdown(
                    id='my_dropdown1',
                    options=[{'label': well_name, 'value': well_name} for well_name in x],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id='my_dropdown2',
                    options=[{'label': feature1, 'value': feature1} for feature1 in y1],
                    multi=False,
                    clearable=False,
                 ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id='my_dropdown3',
                    options=[{'label': feature2, 'value': feature2} for feature2 in y1],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
    ],
    body=True,
)

# Create controls for selecting well and features for a parallel coordinates plot
parallel_controls = dbc.Card(
    [
         html.Div(
            [
                dbc.Label("Select well"),
                dcc.Dropdown(
                    id='my_dropdown11',
                    options=[{'label': well_name, 'value': well_name} for well_name in x],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        
        html.Div(
            [
                dbc.Label("Select Features"),
                dcc.Dropdown(
                    id='dropdown_features',
                    options=[{'label': feature1, 'value': feature1} for feature1 in y1],
                    multi=True,
                    clearable=True,
                 ),
            ]
        ),
    ],
    body=True,
)

# Create controls for selecting well and features for a violin plot
comparison_of_wells = dbc.Card(
    [
         html.Div(
            [
                dbc.Label("Select x axis"),
                dcc.Dropdown(
                    id='my_dropdown_a',
                    options=["Well Name", "Facies"],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        
        html.Div(
            [
                dbc.Label("Select y axis"),
                dcc.Dropdown(
                    id='my_dropdown_b',
                    options=[{'label': feature1, 'value': feature1} for feature1 in y1],
                    multi=False,
                    clearable=False,
                 ),
            ]
        ),
    ],
    body=True,
)
correlation_analysis =  html.Div(
            [
                dbc.Label(" ", id='my_dropdown_y'),         
    ],
)

#Defining page 2 layout
page_2_layout = dbc.Container(
    [
        header1,
        html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="Crossplot", tab_id="scatter"),
                dbc.Tab(label="Parallel Coordinates Plot", tab_id="coord-plot"),
                dbc.Tab(label="Comparison of Wells", tab_id="histogram"),
                dbc.Tab(label="Correlation Analysis", tab_id="heatmap"),
                dbc.Tab(label="Distribution of Facies", tab_id="pie_chart"),
            ],
            id="tabs",
            style={
                "font-size": "16px",
                "border": "none",
                "border-radius": "0",
                "background-color": "#f7f7f7",
                "color": "black",
            },
            #active_tab="scatter",
        ),

        html.Div(id="tab-content", className="p-4"),
    ]
)



@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")],
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        if active_tab == "scatter":
            return dbc.Row(
                [
                    #divides the page layout in two columns, one column for taking input and other for displaying graph
                    dbc.Col(controls, md=4),
                    dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
                ],
                align="center",
            )
        elif active_tab == "coord-plot":
            return dbc.Row(
                [
                    dbc.Col(parallel_controls, md=4),
                    dbc.Col(dcc.Graph(id="parallelplot"), md=8),
                ],
                align="center",
            )
        elif active_tab == "histogram":
            return dbc.Row(
                [
                    dbc.Col(comparison_of_wells, md=4),
                    dbc.Col(dcc.Graph(id="violin_plot"), md=8),
                ],
                align="center",
            )
        elif active_tab == "heatmap":
            return dbc.Row(
                [
                    dbc.Col(correlation_analysis, md=2),
                    dbc.Col(dcc.Graph(id="heatmap"), md=9),
                ],
                align="center",
            )
        elif active_tab == "pie_chart":
            return dbc.Row(
                [
                    dbc.Col(correlation_analysis, md=12),
                    dbc.Col(dcc.Graph(id="pie"), md=10),
                ],
                align="center",
            )
            
    return "No tab selected"



#page 3 layout (Fitting algo)
df = pd.read_csv("assets/facies_vectors.csv")
df=df.dropna()
X=df.iloc[:,3:7]
y=df.iloc[:,8]
X=preprocessing.StandardScaler().fit_transform(X)
y21={'uniform','distance'}
y22={'balanced','balanced_subsample'}
y23={'identity', 'logistic', 'tanh', 'relu'}
y24={'rbf','linear'}



select_well = dbc.Card(
    [
        dbc.CardHeader(
                    html.H4("Select Training Wells", className="text-center")
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select wells"),
                dcc.Dropdown(
                    id='input_wells',
                    options=[{'label': well_name, 'value': well_name} for well_name in x],
                    multi=True,
                    clearable=True,
                 ),
            ]
        ),
        html.Br(),
    ],
    body=True,
)



select_features = dbc.Card(
    [
        dbc.CardHeader(
                    html.H4("Select Input Features", className="text-center")
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select features"),
                dcc.Dropdown(
                    id='input_features',
                    options=[{'label': feature1, 'value': feature1} for feature1 in y1],
                    multi=True,
                    clearable=True,
                 ),
            ]
        ),
        html.Br(),
    ],
    body=True,
)



controls_knn = dbc.Card(
    [
        dbc.CardHeader(
                    html.H4("Select Input Parameters", className="text-center")
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select N-neighbours"),
                dcc.Input(
                    id='k_input',
                    type='number',
                    placeholder='int, e.g. 10',
                    debounce=False, min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Neighbour Weight"),
                dcc.Dropdown(
                    id='knn_dropdown1',
                    placeholder="e.g. 'uniform'",
                    options=[{'label': feature2, 'value': feature2} for feature2 in y21],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Test Size(percentage)"),
                dcc.Input(
                    id='knn_dropdown2',
                    type='number',
                    placeholder='int, e.g. 25',
                    debounce=False, min=0, max=100, step=1,
                    style={'width': '100%'} # set the width to 50% of its container
                ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Model",id="run_model", outline=True, color="primary", className="me-1"),
    ],
    body=True,
)


controls_rf = dbc.Card(
    [
        dbc.CardHeader(
                    html.H4("Select Input Parameters", className="text-center")
        ),
        html.Br(),
      
        html.Div(
             [
                dbc.Label("Select N-estimators"),
                dcc.Input(
                    id='n_estimator',
                    type='number',
                    placeholder='int, e.g. 100',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Minimum Sample Leaf"),
                dcc.Input(
                    id='min_leaf',
                    type='number',
                    placeholder='int, e.g. 5',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Maximum Depth"),
                dcc.Input(
                    id='maxDepth',
                    type='number',
                    placeholder='int, e.g. 10',
                    debounce=False,min=1,
                    style={'width': '100%'}
                    ),
                ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Class Weight"),
                dcc.Dropdown(
                    id='rf_dropdown1',
                    placeholder="e.g. 'balanced'",
                    options=[{'label': feature3, 'value': feature3} for feature3 in y22],
                    #value=y[0],
                    multi=False,
                    clearable=False,
                    #style={"width": "50%"}
                ),
            ]
        ), 
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Test Size(percentage)"),
                dcc.Input(
                    id='rf_dropdown2',
                    type='number',
                    placeholder='int, e.g. 25',
                    debounce=False,min=0,max=100,step=1,
                    style={'width': '100%'} # set the width to 50% of its container
                ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Model",id="run_model1", outline=True, color="primary", className="me-1"),
    ],
    body=True,
)

controls_mlp = dbc.Card(
    [
      dbc.CardHeader(
              html.H4("Select Input Parameters", className="text-center")
      ),
      html.Br(),
      html.Div(
            [
                dbc.Label("Hidden Layer Sizes"),
                dcc.Input(
                    id='hidden_nodes',
                    type='text',
                    placeholder='list of int, e.g. 30,20,....',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Maximum Iteration"),
                dcc.Input(
                    id='max_itr',
                    type='number',
                    placeholder='int, e.g. 100',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Activation Function"),
                dcc.Dropdown(
                    id='mlp_act',
                    options=[{'label': feature4, 'value': feature4} for feature4 in y23],
                    placeholder="e.g.'relu'",
                    #value=y[0],
                    multi=False,
                    clearable=False,
                    #style={"width": "50%"}
                ),
            ]
        ), 
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Test Size(percentage)"),
                dcc.Input(
                    id='mlp_split',
                    type='number',
                    placeholder='int, e.g. 25',
                    debounce=False,min=0,max=100,step=1,
                    style={'width': '100%'} # set the width to 50% of its container
                 ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Model",id="run_model2", outline=True, color="primary", className="me-1"),
    ],
    body=True,
)

controls_svm = dbc.Card(
    [ dbc.CardHeader(
              html.H4("Select Input Parameters", className="text-center")
      ),
      html.Br(),
      html.Div(
            [
                dbc.Label("Select the value of C"),
                dcc.Input(
                    id='s_input',
                    type='number',
                    placeholder='int, e.g. 1',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Choose the value of gamma"),
                dcc.Input(
                    id='s_input1',
                    type='number',
                    placeholder='float, e.g. 0.5',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Choose Kernel"),
                dcc.Dropdown(
                    id='svm_dropdown1',
                    options=[{'label': feature2, 'value': feature2} for feature2 in y24],
                    placeholder="e.g. 'rbf'",
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Test Size(percentage)"),
                dcc.Input(
                    id='svm_dropdown2',
                    type='number',
                    placeholder='int, e.g. 25',
                    debounce=False,min=0,max=100,step=1,
                    style={'width': '100%'} # set the width to 50% of its container
                 ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Model",id="run_model3", outline=True, color="primary", className="me-1"),
    ],
    body=True,
)


controls_xgb = dbc.Card(
    [    dbc.CardHeader(
              html.H4("Select Input Parameters", className="text-center")
      ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select the value of learning rate"),
                dcc.Input(
                    id='x_input2',
                    type='number',
                    placeholder='float, e.g. 0.1',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select the value of max_depth"),
                dcc.Input(
                    id='x_input3',
                    type='number',
                    placeholder='int, e.g. 3',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div([
                dbc.Label("Select the value of alpha"),
                dcc.Input(
                    id='x_input4',
                    type='number',
                    placeholder='int, e.g. 10',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div([
                dbc.Label("Select no of estimators"),
                dcc.Input(
                    id='x_input5',
                    type='number',
                    placeholder='int, e.g. 100',
                    debounce=False,min=0,
                    style={'width': '100%'}
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Test Size(percentage)"),
                dcc.Input(
                    id='x_input6',
                    type='number',
                    placeholder='int, e.g. 25',
                    debounce=False,min=0,max=100,step=1,
                    style={'width': '100%'} # set the width to 50% of its container
                 ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Model",id="run_model4", outline=True, color="primary", className="me-1"),
    ],
    body=True,
)



header2 = html.H2(
    "Fitting Classification Algorithm", className="bg-primary text-white p-2 mb-2 text-center"
)


save_model = dbc.Card(
    [    dbc.CardHeader(
              html.H4("Save model", className="text-center")
      ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Save model as:"),
                dcc.Input(
                    id='save_model_name',
                    type='text',
                    debounce=False,
                    style={'width': '100%'} # set the width to 50% of its container
                 ),
            ]
        ),
         html.Br(),
        dbc.Button("Save",id="save_btn", outline=True, color="primary", className="me-1"),
     dcc.Store(id='store_model', data=[], storage_type='memory')
    ], 
    body=True,
)

# Define the app layout
page_3_layout = dbc.Container([
    header2,
    dbc.Tabs(
            [
                dbc.Tab(label="K-Nearest Neighbor", tab_id="knn"),
                dbc.Tab(label="Random Forest", tab_id="rf"),
                dbc.Tab(label="Multi-Layer Perceptron", tab_id="mlp"),
                dbc.Tab(label="Support Vector Machine", tab_id="svm"),
                dbc.Tab(label="XGBoost", tab_id="gboost"),
            ],
            id="tabs1",
            style={
                "font-size": "16px",
                "border": "none",
                "border-radius": "0",
                "background-color": "#f7f7f7",
                "color": "black",
            },
            #active_tab="scatter",
        ),
    
    html.Div(id="tab-content1", className="p-4"),
    html.Br(),
    save_model,
    html.Br()
])

row1=dbc.Row([
    dbc.Col(controls_knn, md=12)
])

row2=dbc.Row([
    dbc.Col(dcc.Graph(id="knn_cm"), md=5),
    dbc.Col(dcc.Graph(id="knn_f"), md=7)
])

row3=dbc.Row([
    dbc.Col(dcc.Graph(id="knn_cm1"), md=5),
    dbc.Col(dcc.Graph(id="knn_f1"), md=7)
])


rf_row1=dbc.Row([
    dbc.Col(controls_rf, md=12)
])

rf_row2=dbc.Row([
    dbc.Col(dcc.Graph(id="rf_cm"), md=5),
    dbc.Col(dcc.Graph(id="rf_f"), md=7)
])

rf_row3=dbc.Row([
    dbc.Col(dcc.Graph(id="rf_cm1"), md=5),
    dbc.Col(dcc.Graph(id="rf_f1"), md=7)
])



mlp_row1=dbc.Row([
    dbc.Col(controls_mlp, md=12)
])

mlp_row2=dbc.Row([
    dbc.Col(dcc.Graph(id="mlp_cm"), md=5),
    dbc.Col(dcc.Graph(id="mlp_f"), md=7)
])

mlp_row3=dbc.Row([
    dbc.Col(dcc.Graph(id="mlp_cm1"), md=5),
    dbc.Col(dcc.Graph(id="mlp_f1"), md=7)
])

svm_row1=dbc.Row([
    dbc.Col(controls_svm, md=12)
])

svm_row2=dbc.Row([
    dbc.Col(dcc.Graph(id="svm_cm"), md=5),
    dbc.Col(dcc.Graph(id="svm_f"), md=7)
])

svm_row3=dbc.Row([
    dbc.Col(dcc.Graph(id="svm_cm1"), md=5),
    dbc.Col(dcc.Graph(id="svm_f1"), md=7)
])

xgb_row1=dbc.Row([
    dbc.Col(controls_xgb, md=12)
])

xgb_row2=dbc.Row([
    dbc.Col(dcc.Graph(id="xgb_cm"), md=5),
    dbc.Col(dcc.Graph(id="xgb_f"), md=7)
])

xgb_row3=dbc.Row([
    dbc.Col(dcc.Graph(id="xgb_cm1"), md=5),
    dbc.Col(dcc.Graph(id="xgb_f1"), md=7)
])



@app.callback(
    Output("tab-content1", "children"),
    [Input("tabs1", "active_tab")],
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab is not None:
        if active_tab == "knn":
            return dbc.Col(
                [
                   row1,
                    dbc.CardHeader(
                    html.H4("Performance on training data", className="text-center"),
        ),row2,
                    dbc.CardHeader(
                    html.H4("Performance on validation data", className="text-center")
        ),row3
                ]
            )
        elif active_tab == "rf":
            return dbc.Col(
                [
                   rf_row1,
                    dbc.CardHeader(
                    html.H4("Performance on training data", className="text-center")
        ),rf_row2,
                    dbc.CardHeader(
                    html.H4("Performance on validation data", className="text-center")
        ),rf_row3
                ]
            )
        elif active_tab == "mlp":
            return dbc.Col(
                [
                   mlp_row1,
                    dbc.CardHeader(
                    html.H4("Performance on training data", className="text-center")
        ),mlp_row2,
                    dbc.CardHeader(
                    html.H4("Performance on validation data", className="text-center")
        ),mlp_row3
                ]
            )
        elif active_tab == "svm":
            return dbc.Col(
                [
                   svm_row1,
                    dbc.CardHeader(
                    html.H4("Performance on training data", className="text-center")
        ),svm_row2,
                    dbc.CardHeader(
                    html.H4("Performance on validation data", className="text-center")
        ),svm_row3
                ]
            )
        elif active_tab == "gboost":
            return dbc.Col(
                [
                   xgb_row1,
                    dbc.CardHeader(
                    html.H4("Performance on training data", className="text-center")
        ),xgb_row2,
                    dbc.CardHeader(
                    html.H4("Performance on validation data", className="text-center")
        ),xgb_row3
                ]
            )
            
    return "No tab selected"






#page 4 layout

# Page showing the saved model and using it to predict facies

df = pd.read_csv("assets/facies_vectors.csv")
x=df['Well Name'].unique()
x=list(x)

header3 = html.H2(
    "Predict with saved model", className="bg-primary text-white p-2 mb-2 text-center"
)
controls_predict = dbc.Card(
    [
        dbc.CardHeader(
                    html.H4("Select Parameters", className="text-center")
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Well"),
                dcc.Dropdown(
                    id='predict_dd1_well',
                    options=[{'label': well_name, 'value': well_name} for well_name in x],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        html.Br(),
        html.Div(
            [
                dbc.Label("Select Model"),
                dcc.Dropdown(
                    id='predict_dd2_model',
                    options=[],
                    multi=False,
                    clearable=False,
                ),
            ]
        ),
        html.Br(),
        dbc.Button("Run Algorithm",id="run_btn", outline=True, color="primary",className="d-grid gap-2",),
    ],
    body=True,
)


page_4_layout=dbc.Container(
    [
        header3,
        html.Br(),
        dbc.Row([
            
            dbc.Col(controls_predict, md=4),
            dbc.Col(dcc.Graph(id="predict_plot"), md=8),
        ],
        align="center",
        ),

        html.Div(id="tab-content", className="p-4"),
    ]
)








#page_1_callback

#Code to import the data in .las,.csv format

def generate_well_plot(wdd):
    #wdd1=wdd[wdd['Well Name']=='M-5']
    wdd1=wdd
    Att1=['DTS', 'RESIS',  'DT', 'NPHI', 'GR', 'RHOB']
    fig = make_subplots(rows=1, cols=len(Att1)+1, shared_yaxes=True, horizontal_spacing=0.005,
                        subplot_titles=(np.append(Att1,['Facies'])), )
    
#     subplot_titles=(np.append(Att1,['Facies'])), )

    for i in range(len(Att1)):
        fig.add_trace(go.Scatter(x=wdd1[Att1[i]], y=wdd1.Depth),
                      row=1, col=i+1)

    z=np.repeat(np.expand_dims(wdd1.LITH.values,1), 1, 1)
    

    d_cmap=[[0/7,'navajowhite'], [1/7,'navajowhite'], 
            [1/7,'cornflowerblue'], [2/7,'cornflowerblue'], 
            [2/7,'seagreen'], [3/7,'seagreen'], 
            [3/7,'saddlebrown'], [4/7,'saddlebrown'],
            [4/7,'skyblue'], [5/7,'skyblue'],
            [5/7,'yellowgreen'], [6/7,'yellowgreen'], 
            [6/7,'peru'], [7/7,'peru'], ]

    fig.add_trace(go.Heatmap(y=wdd1.Depth, z = z, zmin=0, zmax=6, type = 'heatmap', 
                             colorscale=d_cmap, colorbar = dict(thickness=15)), 
                  row=1, col=i+2)



    fig.update_layout(height=700,title_text="Well logs plot with depth", showlegend=False)
    #height=400, width=400 ,

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(font_size=15, font_color='black')
    return fig



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'las' in filename:
            # Assume that the user uploaded a LAS file
            las = lasio.read(io.StringIO(decoded.decode('utf-8')))
            df = las.df()
            df=df.dropna()
            print(df)
    except Exception as e:
        return html.Div([
            'Invalid file type.'
        ])

    return html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=15
        ),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
    ])


# Implementing well plot for the imported data


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
        #children.to_csv('my_df.csv', index=False) 
    return children


@app.callback(
    Output('well_plot', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_well_plot(contents, filename):
    if contents:
        # Parse the uploaded file into a Pandas dataframe
        content_type, content_string = contents[0].split(',')
        decoded = base64.b64decode(content_string)
        las = lasio.read(io.StringIO(decoded.decode('utf-8')))
        wdd = las.df()
        wdd['Depth']=wdd.index
        wdd=wdd.dropna()
        print(wdd)
        # Generate the well plot using the dataframe
        fig = generate_well_plot(wdd)
        return fig

    # If no file is uploaded yet, display an empty plot
    return {}
@app.callback(
    Output("example-output", "children"), [Input("add_well", "n_clicks")]
)
def on_button_click(n):
    if not n or n % 2 == 0:
        return {}
    else:
        return "Well Added."







#page_2_callback

# Implementing heatmap showing correlation among features and useful for outlier detection

@app.callback(
    Output(component_id='cluster-graph', component_property='figure'),
    [Input(component_id='my_dropdown1', component_property='value'),
    Input(component_id='my_dropdown2', component_property='value'),
    Input(component_id='my_dropdown3', component_property='value')])

def update_graph(my_dropdown1,my_dropdown2,my_dropdown3):
    df=pd.read_csv("assets/facies_vectors.csv")
    dff = df[df['Well Name']==my_dropdown1]
    k=dff["Facies"].astype(str)
    fig = px.scatter(dff, x=my_dropdown2, y=my_dropdown3, color=k,title="Crossplot",
                 hover_data=['Depth','GR'], width=700, height=500)
                 
    return fig

# Implementing parallel plot giving patterns and relationship between variables

@app.callback(
    Output(component_id='parallelplot', component_property='figure'),
    [Input(component_id='dropdown_features', component_property='value'),
    Input(component_id='my_dropdown11', component_property='value')])
    #Input(component_id='my_dropdown4', component_property='value')]

def update_graph1(dropdown_features,my_dropdown11):
    df1=pd.read_csv("assets/facies_vectors.csv")
    if my_dropdown11 == "ALL WELLS":
        dff=df1
    else :
        dff = df1[df1['Well Name']==my_dropdown11]
    k1=dff["Facies"].astype(str)
    fig1 = px.parallel_coordinates(dff,color='Facies',
                              dimensions=dropdown_features,
                              color_continuous_scale=px.colors.diverging.Tealrose)
                 
    return fig1

# Implementing violing plot giving summary statistics and distribution of data and useful for outlier 

@app.callback(
    Output(component_id='violin_plot', component_property='figure'),
    [Input(component_id='my_dropdown_a', component_property='value'),
     Input(component_id='my_dropdown_b', component_property='value')])
    #Input(component_id='my_dropdown4', component_property='value')] 

def update_graph2(my_dropdown_a, my_dropdown_b):
    df2=pd.read_csv("assets/facies_vectors.csv")
    fig2 = px.violin(df2 ,
                     x=my_dropdown_a , y=my_dropdown_b )
                 
    return fig2

# Implementing heatmap showing correlation among features

@app.callback(
    Output(component_id='heatmap', component_property='figure'),
    [Input(component_id='my_dropdown_y', component_property='value')])

def update_graph4(my_dropdown_y):
    df4 = pd.read_csv("assets/facies_vectors.csv")
    df4.head()
    x=df4['Well Name'].unique()
    x=list(x)
    y1=df4.columns[1:7]
    cor_mat=df4[y1].corr()
    cor_mat=np.round(cor_mat, 2)
    fig4 = px.imshow(cor_mat,height=700,width=700, text_auto=True, color_continuous_scale=px.colors.sequential.RdBu, )
                 
    return fig4

#Implementing piechart demonstrating the distribution of classes

@app.callback(
    Output(component_id='pie', component_property='figure'),
    [Input(component_id='my_dropdown_y', component_property='value')])

def update_graph5(my_dropdown_y):
    wd=pd.read_csv("assets/facies_vectors.csv")
    u,c = np.unique(wd.Facies, return_counts=True)
    c = c/np.sum(c)
    fig5=go.Figure(data=go.Pie(labels=u, values=c))
    return fig5




with open('clf_list.pkl', 'wb') as f:
    pickle.dump([], f)




#page_3_callback

# Define the app callbacks
#callback for knn:
@app.callback(
    [Output("knn_cm", "figure"), Output("knn_f", "figure"),
     Output("knn_cm1", "figure"), Output("knn_f1", "figure"),
     Output('run_model', 'n_clicks')],
    [   Input('run_model', 'n_clicks'),
        Input("k_input", "value"),
        Input("knn_dropdown1", "value"),
        Input("knn_dropdown2", "value"),
    ],
)

def update_knn_cm(n_clicks, k_input, knn_dropdown1, knn_dropdown2):
    if n_clicks is not None:
            knn = KNeighborsClassifier()
            knn.set_params(n_neighbors=k_input,weights=knn_dropdown1)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=knn_dropdown2/100,random_state=1)
            knn.fit(X_train, y_train)
            with open('clf.pkl', 'wb') as f:
                pickle.dump(knn, f)
            y_pred = knn.predict(X_train)
            cm = confusion_matrix(y_train, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig=fig.update_layout(title='Confusion Matrix')   #Returning the confusion matrix for training data
            
            crept=classification_report(y_train, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig1=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig1=fig1.update_layout(title='Performance metrics') #Returning the classificatiobn report for training data
            
            y_pred = knn.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig2 = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig2=fig2.update_layout(title='Confusion Matrix')  #Returning the confusion matrix for test data
            
            crept=classification_report(y_test, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig3=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig3=fig3.update_layout(title='Performance metrics') #Returning the classificatiobn report for test data

            return fig, fig1, fig2, fig3, 0


# Implementing the Random Forest
      
        
@app.callback(
    [Output("rf_cm", "figure"), Output("rf_f", "figure"),
     Output("rf_cm1", "figure"), Output("rf_f1", "figure"),
     Output('run_model1', 'n_clicks')],
    [   Input('run_model1', 'n_clicks'),
        Input("n_estimator", "value"),
        Input("min_leaf", "value"),
        Input("maxDepth", "value"),
        Input("rf_dropdown1", "value"),
        Input("rf_dropdown2", "value"),
    ],
)

def update_rf_cm(n_clicks, n_estimator, min_leaf, maxDepth, rf_dropdown1, rf_dropdown2):
    if n_clicks > 0:
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=rf_dropdown2/100,random_state=1)
            clf = RandomForestClassifier(n_estimators=n_estimator,max_depth=maxDepth,
                                         min_samples_split=min_leaf,class_weight=rf_dropdown1)
            clf.fit(X_train, y_train)
            with open('clf.pkl', 'wb') as f:
                pickle.dump(clf, f)
            y_pred = clf.predict(X_train)
            cm = confusion_matrix(y_train, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig=fig.update_layout(title='Confusion Matrix') #Returning the confusion matrix for training data
            
            crept=classification_report(y_train, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig1=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig1=fig1.update_layout(title='Performance metrics') #Returning the classificatiobn report for training data
            
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig2 = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig2=fig2.update_layout(title='Confusion Matrix') #Returning the confusion matrix for test data 
            
            
            crept=classification_report(y_test, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig3=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig3=fig3.update_layout(title='Performance metrics') #Returning the classificatiobn report for test data
            
            return fig, fig1, fig2, fig3, 0



# Implementing the Multi Layer Perceptron


@app.callback(
    [Output("mlp_cm", "figure"), Output("mlp_f", "figure"),
     Output("mlp_cm1", "figure"), Output("mlp_f1", "figure"),
     Output('run_model2', 'n_clicks')],
    [   Input('run_model2', 'n_clicks'),
        Input("hidden_nodes", "value"),
        Input("max_itr", "value"),
        Input("mlp_act", "value"),
        Input("mlp_split", "value"),
    ],
)

def update_mlp_cm(n_clicks, hidden_nodes, max_itr, mlp_act, mlp_split):
    if n_clicks > 0:
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=mlp_split/100,random_state=1)
            classifier = MLPClassifier(hidden_layer_sizes=eval(hidden_nodes), max_iter=max_itr,activation = mlp_act)
            classifier.fit(X_train, y_train)
            with open('clf.pkl', 'wb') as f:
                pickle.dump(classifier, f)
            y_pred = classifier.predict(X_train)
            cm = confusion_matrix(y_train, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig=fig.update_layout(title='Confusion Matrix') #Returning the confusion matrix for training data
            
            crept=classification_report(y_train, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig1=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig1=fig1.update_layout(title='Performance metrics') #Returning the classificatiobn report for training data
            
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig2 = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig2=fig2.update_layout(title='Confusion Matrix')  #Returning the confusion matrix for test data 
            
            crept=classification_report(y_test, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig3=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig3=fig3.update_layout(title='Performance metrics')  #Returning the classificatiobn report for test data
            return fig, fig1, fig2, fig3, 0
        

# Implementing the Support Vector Machine
        
@app.callback(
    [Output("svm_cm", "figure"), Output("svm_f", "figure"),
     Output("svm_cm1", "figure"), Output("svm_f1", "figure"),
     Output('run_model3', 'n_clicks')],
    [   Input('run_model3', 'n_clicks'),
        Input("s_input", "value"),
        Input("s_input1", "value"),
        Input("svm_dropdown1", "value"),
        Input("svm_dropdown2", "value"),
    ],
)

def update_svm_cm(n_clicks, s_input, s_input1, svm_dropdown1, svm_dropdown2):
    if n_clicks > 0:
        if svm_dropdown1=='linear':
            model = SVC(kernel=svm_dropdown1, C=s_input)
        else:
            model = SVC(kernel=svm_dropdown1, C=s_input,gamma=s_input1) #If user chooses rbf kernel ,we use gamma
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=svm_dropdown2/100,random_state=1)
        model.fit(X_train, y_train)
        with open('clf.pkl', 'wb') as f:
            pickle.dump(model, f)
        y_pred = model.predict(X_train)
        cm = confusion_matrix(y_train, y_pred, )
        cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        cm_normalized=np.round(cm_normalized,2)
        fig = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
        fig=fig.update_layout(title='Confusion Matrix') #Returning the confusion matrix for training data
        
        crept=classification_report(y_train, y_pred,output_dict=True)
        df = pd.DataFrame(crept).transpose()
        fig1=px.bar(df.iloc[:-3,:-1], barmode='group')
        fig1=fig1.update_layout(title='Performance metrics') #Returning the classificatiobn report for training data

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, )
        cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        cm_normalized=np.round(cm_normalized,2)
        fig2 = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
        fig2=fig2.update_layout(title='Confusion Matrix')  #Returning the confusion matrix for test data


        crept=classification_report(y_test, y_pred,output_dict=True)
        df = pd.DataFrame(crept).transpose()
        fig3=px.bar(df.iloc[:-3,:-1], barmode='group')
        fig3=fig3.update_layout(title='Performance metrics')  #Returning the classificatiobn report for test data
        return fig, fig1, fig2, fig3, 0
        


#Implementing XGBoost Algorithm

@app.callback(
    [Output("xgb_cm", "figure"), Output("xgb_f", "figure"),
     Output("xgb_cm1", "figure"), Output("xgb_f1", "figure"),
     Output('run_model4', 'n_clicks')],
    [   Input('run_model4', 'n_clicks'),
        Input("x_input2", "value"),
        Input("x_input3", "value"),
        Input("x_input4", "value"),
        Input("x_input5", "value"),
        Input("x_input6","value")
    ],
)

def update_xgboost_cm(n_clicks, x_input2, x_input3, x_input4, x_input5, x_input6):
    if n_clicks > 0:
            model = xgb.XGBClassifier(objective ='reg:linear',learning_rate = x_input2,
                                      max_depth = x_input3, alpha = x_input4, n_estimators = x_input5)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=x_input6/100,random_state=1)
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
            model.fit(X_train, y_train)
            with open('clf.pkl', 'wb') as f:
                pickle.dump(model, f)
            y_pred = model.predict(X_train)
            cm = confusion_matrix(y_train, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig=fig.update_layout(title='Confusion Matrix')  #Returning the confusion matrix for training data
            
            crept=classification_report(y_train, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig1=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig1=fig1.update_layout(title='Performance metrics') #Returning the classificatiobn report for training data

            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, )
            cm_normalized=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
            cm_normalized=np.round(cm_normalized,2)
            fig2 = px.imshow(cm_normalized, labels=dict(x="Predicted", y="Actual", color="Count"),text_auto=True,aspect="auto")
            fig2=fig2.update_layout(title='Confusion Matrix') #Returning the confusion matrix for test data

            crept=classification_report(y_test, y_pred,output_dict=True)
            df = pd.DataFrame(crept).transpose()
            fig3=px.bar(df.iloc[:-3,:-1], barmode='group')
            fig3=fig3.update_layout(title='Performance metrics')  #Returning the classificatiobn report for test data
            return fig, fig1, fig2, fig3, 0
        
        

        
#Saving the trained model to predict the facies in the next pag
        
@app.callback(
    [Output("store_model", "data"), Output('save_btn', 'n_clicks'), ],
    [Input('save_btn', 'n_clicks'),
     Input("save_model_name", "value"),
#      Input("store_model", "data")
    ],
)

    
def save_model(n_clicks, name):
    with open('clf.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('clf_list.pkl', 'rb') as f:
        clf_list = pickle.load(f)
    if n_clicks > 0:
        with open(name+'.pkl', 'wb') as f:
            pickle.dump(clf, f)
        clf_list.append(name)
        print(clf_list)
#         for clf in clf_list:
#             data.append(clf)
#         data.append(clf_list)
        clf_list=list(set(clf_list))
        with open('clf_list.pkl', 'wb') as f:
            pickle.dump(clf_list, f)
    return clf_list, 0
        




#page_4_callback

def generate_well_plot1(wdd, y):
    #wdd1=wdd[wdd['Well Name']=='M-5']
    wdd1=wdd
    Att1=['DTS', 'RESIS',  'DT', 'NPHI', 'GR', 'RHOB']
    fig = make_subplots(rows=1, cols=len(Att1)+1, shared_yaxes=True, horizontal_spacing=0.005,
                        subplot_titles=(np.append(Att1,['Pred Facies'])), )
    

    for i in range(len(Att1)):
        fig.add_trace(go.Scatter(x=wdd1[Att1[i]], y=wdd1.Depth),
                      row=1, col=i+1)

    z=np.repeat(np.expand_dims(y,1), 1, 1)
    

    d_cmap=[[0/7,'navajowhite'], [1/7,'navajowhite'], 
            [1/7,'cornflowerblue'], [2/7,'cornflowerblue'], 
            [2/7,'seagreen'], [3/7,'seagreen'], 
            [3/7,'saddlebrown'], [4/7,'saddlebrown'],
            [4/7,'skyblue'], [5/7,'skyblue'],
            [5/7,'yellowgreen'], [6/7,'yellowgreen'], 
            [6/7,'peru'], [7/7,'peru'], ]

    fig.add_trace(go.Heatmap(y=wdd1.Depth, z = z, zmin=0, zmax=6, type = 'heatmap', 
                             colorscale=d_cmap, colorbar = dict(thickness=15)), 
                  row=1, col=i+2)



    fig.update_layout(height=700,title_text="Well logs plot with depth", showlegend=False)
    #height=400, width=400 ,

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(font_size=15, font_color='black')
    return fig



# Selecting the saved model to predict the facies for the test data
@app.callback(
    [Output("predict_dd2_model", "options")],
    [Input('predict_dd1_well', 'value'),
     Input("predict_dd2_model", "options")
     ],
)

def update_model_list(value, options):
    if value is not None:
        with open('clf_list.pkl', 'rb') as f:
            clf_list = pickle.load(f)
        options.append([{'label': clf, 'value': clf} for clf in clf_list])
        return options

#Genrating the well plot predficting facies in the 4th page
@app.callback(
    [Output("predict_plot", "figure"), Output('run_btn', 'n_clicks'),],
    [Input('run_btn', 'n_clicks'),
     Input('predict_dd1_well', 'value'),
     Input("predict_dd2_model", "value")
     ],
)

def plot(n_clicks, value, clf_name):
    if n_clicks:
        df=pd.read_csv("assets/facies_vectors.csv")
        wdd=df[df['Well Name']==value]
        X=wdd.iloc[:,3:7]
        X=preprocessing.StandardScaler().fit_transform(X)
        with open(clf_name+'.pkl', 'rb') as f:
            clf = pickle.load(f)
        y=clf.predict(X)
        
        return generate_well_plot1(wdd, y), 0
    


# main callback
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return home_layout #html.P("This is the content of the home page!")
    elif pathname == "/page-1":
        return page_1_layout #html.P("This is the content of page 1,where we are importing the data")
    elif pathname == "/page-2":
        return page_2_layout #html.P("This is the content of page 2,where we are visulising the input data")
    elif pathname == "/page-3":
        return page_3_layout #html.P("This is the content of page 3,where we are fitting the ML model to the data")
    elif pathname == "/page-4":
        return page_4_layout #html.P("This is the content of page 4,where we are predicting the facies")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )







if __name__ == "__main__":
    app.run_server(debug=True)