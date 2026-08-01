import urllib.request
import zipfile
import io
import pandas as pd

def test_fetch():
    print("Fetching Fama-French 5-Factor daily data...")
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
        print("Zip files found:", zip_file.namelist())
        with zip_file.open(zip_file.namelist()[0]) as f:
            content = f.read().decode('utf-8')
            
    # Inspect first 10 lines
    lines = content.split('\n')
    print("First 10 lines:")
    for idx, line in enumerate(lines[:10]):
        print(f"  {idx}: {line}")
        
    # Find row with headers
    skip_rows = 0
    for i, line in enumerate(lines[:15]):
        if 'Mkt-RF' in line:
            skip_rows = i
            break
            
    print(f"Dynamic skip rows detected: {skip_rows}")
    df = pd.read_csv(io.StringIO(content), skiprows=skip_rows)
    print("Columns read:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head(5))

if __name__ == "__main__":
    test_fetch()
