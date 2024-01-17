import plotly.express as px

from dash import Dash, html, dcc, Input, Output

class PlotDash:
    def __init__(self,
                 data=None):

        '''
        data (pd.DataFrame: columns=['daily_cos_sim', 'daily_headline', 'daily_article']): Pandas dataframe to be plotted
        jupyter (bool): Flag to indicate if running in a Jupyter notebook
        '''

        self.data = data
        self.app = Dash(__name__)
        self._setup_app()

    def _setup_app(self):
        # Create a Plotly Express line graph
        fig = px.line(self.data, y='daily_cos_sim', title='Daily Cosine Similarity Over Time')

        # Update layout for font
        fig.update_layout(
            title_font=dict(family='Helvetica', size=21),
            font=dict(family='Helvetica', size=11),
            plot_bgcolor='white',
            paper_bgcolor='white',
        )

        # Update axis titles with bold font
        fig.update_xaxes(title_text='Date',
                         title_font=dict(family='Helvetica', size=18),
                         showgrid=False,
                         showline=True,
                         linewidth=1,
                         linecolor='black',
                         type='category'
                         )
        fig.update_yaxes(title_text='Cosine Similarity',
                         title_font=dict(family='Helvetica', size=18),
                         showgrid=False,
                         showline=True,
                         linewidth=1,
                         linecolor='black'
                         )

        # Initialize Dash App Plot
        self.app.layout = html.Div([
            dcc.Graph(
                id='cos_sim_graph',
                figure=fig
            ),
            html.Div(
                id='text-output',
                style={'font-family': 'Helvetica', 'padding': '50px'}
            )
        ])

        # Callback to update the text-output div
        @self.app.callback(
            Output('text-output', 'children'),
            Input('cos_sim_graph', 'clickData')
        )
        def display_click_data(clickData):
            if clickData is not None:
                # Extract the date of the clicked point
                clicked_date = clickData['points'][0]['x']
                # Retrieve the headline and article for the clicked date
                selected_data = self.data.loc[self.data.index == clicked_date, ['daily_headline', 'daily_document']].iloc[0]
                return html.Div([
                    html.Div(
                        selected_data['daily_headline'],
                        style={'font-family': 'Helvetica', 'font-weight': 'bold', 'font-size': '20px', 'margin-bottom': '20px', 'line-height': '1.5', 'padding_left': '250px', 'padding_right': '250px'}
                    ),
                    html.Div(
                        selected_data['daily_document'],
                        style={'font-family': 'Helvetica', 'font-weight': 'normal', 'font-size': '14px', 'line-height': '1.5', 'padding_left': '250px', 'padding_right': '250px'}
                    )
                ])

    def plot_dash(self, debug=True):
        self.app.run_server(debug=debug)