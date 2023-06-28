import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1,cols=2,subplot_titles=('First plot','Second plot'),
                   specs=[[{'type': 'scene'}, {'type': 'scene'}]])

fig.add_trace(go.Scatter3d(x=[0,1,2,3],y=[0,1,2,3],z=[0,1,2,3]), row=1,col=1)
fig.add_trace(go.Scatter3d(x=[0,1,2,3],y=[0,1,2,3], z=[0,1,2,3]), row=1,col=2)

fig.update_layout(title='Add Custom Data')
fig.add_trace(go.Scatter3d(x=[0,1,2,3],y=[0,1,2,3],z=[0,1,2,3],
text=['print("Hello World!")','import os','import sys','z=1+2/3'], hovertemplate='%{text}'), row=1,col=1)
fig.show()