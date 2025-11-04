import pandas as pd
seasons = ["20232024", "20222023", "20212022", "20202021", "20242025"]
for season in seasons:
    fileName = f"{season}GameData.csv"
    df = pd.read_csv(f"rawData/{fileName}")
    sorted_df = df.sort_values(by='date', ascending=True)
    sorted_df.to_csv(f'sortedData/Sorted{fileName}', index=False)
