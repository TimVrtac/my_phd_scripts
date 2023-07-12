# odpiranje pd.DataFrame-ov v novem oknu
import webbrowser
import pandas as pd
from tempfile import NamedTemporaryFile

def df_window(df):
    with NamedTemporaryFile(mode='r+', delete=False, suffix='.html') as f:
        df.to_html(f)
    webbrowser.open(f.name)