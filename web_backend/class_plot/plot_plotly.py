import plotly.express as px
import json
import plotly

class PlotPlotly:
    def __init__(self,
                 data=None):

        '''
        data (pd.DataFrame): Pandas DataFrame to be plotted with columns: ['daily_cos_sim', 'daily_headline', 'daily_article']
        '''
        self.data = data

    def _generate_plot(self):
        # Create a Plotly Express line graph
        fig = px.line(self.data)

        # Update layout for font
        fig.update_layout(
            title_font=dict(family='Helvetica', size=16),
            font=dict(family='Helvetica', size=10),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )

        # Update axis
        fig.update_xaxes(title_text='Date',
                         title_font=dict(family='Helvetica', size=13),
                         title_standoff=20,
                         showgrid=False,
                         showline=True,
                         linewidth=1,
                         linecolor='black',
                         type='category',
                         tickangle=45,
                         tickfont=dict(size=10))
        fig.update_yaxes(title_text='Attention',
                         title_font=dict(family='Helvetica', size=13),
                         title_standoff=20,
                         showgrid=False,
                         showline=True,
                         linewidth=1,
                         linecolor='black',
                         tickfont=dict(size=10))

        return fig

    def get_plot(self):
        fig = self._generate_plot()
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
