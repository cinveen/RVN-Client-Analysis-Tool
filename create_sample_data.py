import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure the Sample_Data directory exists
os.makedirs('Sample_Data', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters for the sample data
num_rows = 1000
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 7, 22)
channels = ['TV5', 'TV5 Monde']
markets = ['France']

# Define sample story slugs based on the real data
story_slugs = [
    'NIGER-SECURITY/',
    'BURKINA-SECURITY/',
    'CONGO-SECURITY/',
    'SENEGAL-ELECTION/',
    'SENEGAL-POLITICS/',
    'MALI-SECURITY/',
    'FRANCE-POLITICS/',
    'AFRICA-SUMMIT/',
    'FRANCE-ECONOMY/',
    'EU-FRANCE/',
    'FRANCE-PROTESTS/',
    'FRANCE-ELECTION/',
    'CLIMATE-CHANGE/',
    'COVID-FRANCE/',
    'UKRAINE-CRISIS/'
]

# Generate random dates between start and end date
def random_dates(start, end, n):
    delta = end - start
    return [start + timedelta(seconds=np.random.randint(0, int(delta.total_seconds()))) for _ in range(n)]

# Generate sample data
data = {
    'Channel: Name': np.random.choice(channels, num_rows, p=[0.55, 0.45]),  # 55% TV5, 45% TV5 Monde
    'Market: Name': np.random.choice(markets, num_rows),
    'UTC detection start': random_dates(start_date, end_date, num_rows),
    'Story ID': [f'S{np.random.randint(1000, 9999)}' for _ in range(num_rows)],
    'Slug line': np.random.choice(story_slugs, num_rows, p=[0.15, 0.10, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
    'Hit: EID': [f'EID{np.random.randint(10000, 99999)}' for _ in range(num_rows)],
    'Duration (secs)': np.random.randint(30, 300, num_rows),
    'Headline': [f'Sample headline for story {i+1}' for i in range(num_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add derived columns
df['Local detection start'] = df['UTC detection start'] + timedelta(hours=2)  # Assuming France is UTC+2

# Generate random days for activation dates
random_days = np.random.randint(1, 30, num_rows)
df['Hit: Activation date start (UTC)'] = [dt - timedelta(days=int(days)) for dt, days in zip(df['UTC detection start'], random_days)]
df['Asset: Activation date start (UTC)'] = df['Hit: Activation date start (UTC)']

# Generate detection lengths (2-55 seconds)
detection_seconds = np.random.randint(2, 56, num_rows)
df['Actual detection length'] = [timedelta(seconds=int(s)) for s in detection_seconds]

# Save to Excel
output_file = 'Sample_Data/Sample_Teletrax_Data.xlsx'
df.to_excel(output_file, index=False)

print(f"Sample data created with {num_rows} rows and saved to {output_file}")
print(f"Date range: {df['UTC detection start'].min()} to {df['UTC detection start'].max()}")
print(f"Channels: {', '.join(df['Channel: Name'].unique())}")
print(f"Number of unique Story IDs: {df['Story ID'].nunique()}")
print(f"Number of unique Slug lines: {df['Slug line'].nunique()}")
