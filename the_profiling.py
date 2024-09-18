import pandas as pd
from ydata_profiling import ProfileReport
from IPython.core.display import display

df = pd.read_csv('C:\\Users\\VTOLW\\Downloads\\iris\\bezdekIris.csv')
profile = ProfileReport(df, title="Reporte")

df.head()

profile.to_notebook_iframe()

profile.to_file("reporte.html")
