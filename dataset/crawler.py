import requests
import pandas as pd
from urllib.parse import urlencode
from io import StringIO
from typing import List, Optional


'''Global var'''
# URLs and API using exoplanet archive
BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"
API=''

#columns
COLUMNS = [
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_period",
    "koi_period_err1",
    "koi_period_err2",
    "koi_time0bk",
    "koi_time0bk_err1",
    "koi_time0bk_err2",
    "koi_impact",
    "koi_impact_err1",
    "koi_impact_err2",
    "koi_duration",
    "koi_duration_err1",
    "koi_duration_err2",
    "koi_depth",
    "koi_depth_err1",
    "koi_depth_err2",
    "koi_prad",
    "koi_prad_err1",
    "koi_prad_err2",
    "koi_teq",
    "koi_teq_err1",
    "koi_teq_err2",
    "koi_insol",
    "koi_insol_err1",
    "koi_insol_err2",
    "koi_model_snr",
    "koi_tce_plnt_num",
    "koi_tce_delivname",
    "koi_steff",
    "koi_steff_err1",
    "koi_steff_err2",
    "koi_slogg",
    "koi_slogg_err1",
    "koi_slogg_err2",
    "koi_srad",
    "koi_srad_err1",
    "koi_srad_err2",
    "ra",
    "dec",
    "koi_kepmag",
]




'''Functions'''



#Build the API URL with query parameters
def build_api_url(columns: List[str], table: str = "cumulative"):
    params = {
        "table": table,
        "select": ",".join(columns),
        "format": "csv",
    }
    return f"{BASE_URL}?{urlencode(params)}"




#Fetch data from the NASA API
def fetch_exoplanet_data(url: str, timeout: int = 30):
    print("Requesting:", url)
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        raise SystemExit(f"Failed to fetch data from Exoplanet Archive API: {e}")





#Convert CSV text to DataFrame and reorder columns
def process_csv(csv_text: str, columns: List[str]) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    return df.reindex(columns=columns)





#save
def save(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)
    print(f"Saved {len(df):,} rows to {filename}")





def main(output_file: str = "exoplanet.csv", 
                        columns: Optional[List[str]] = None):

    if columns is None:
        columns = COLUMNS
    
    url = build_api_url(columns)
    csv_text = fetch_exoplanet_data(url)
    df = process_csv(csv_text, columns)
    save(df, output_file)

if __name__ == "__main__":
    main("exoplanet.csv")
