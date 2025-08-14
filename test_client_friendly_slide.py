import os
import pandas as pd
import matplotlib.pyplot as plt
from teletrax_analysis import create_single_slide_presentation, timedelta_to_seconds

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load the data
print("Loading data...")
try:
    file_path = 'Sample_Data/Channel News Asia_2025_08_05_12_43_57.xlsx'
    df = pd.read_excel(file_path)
    print(f"Loaded actual Teletrax data from {file_path}")
except FileNotFoundError:
    try:
        file_path = 'Sample_Data/TV5 test_2025_07_23_18_19_36.xlsx'
        df = pd.read_excel(file_path)
        print(f"Loaded actual Teletrax data from {file_path}")
    except FileNotFoundError:
        try:
            file_path = 'Sample_Data/Sample_Teletrax_Data.xlsx'
            df = pd.read_excel(file_path)
            print(f"Loaded sample data from {file_path}")
        except FileNotFoundError:
            print("No data file found. Please run create_sample_data.py to generate sample data or place your Teletrax export in the Sample_Data directory.")
            exit(1)

# Process the data
print("Processing data...")
# Convert datetime columns to datetime type
datetime_cols = ['UTC detection start', 'Local detection start', 
                'Hit: Activation date start (UTC)', 'Asset: Activation date start (UTC)']
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Convert 'Actual detection length' to seconds for easier analysis
df['Detection Length (seconds)'] = df['Actual detection length'].apply(timedelta_to_seconds)

# Add derived columns for analysis
df['Detection Year'] = df['UTC detection start'].dt.year
df['Detection Month'] = df['UTC detection start'].dt.month
df['Detection Day'] = df['UTC detection start'].dt.day
df['Detection Hour'] = df['UTC detection start'].dt.hour
df['Detection Weekday'] = df['UTC detection start'].dt.day_name()
df['Detection Date'] = df['UTC detection start'].dt.date

# Set the df variable in teletrax_analysis to our loaded data
import teletrax_analysis
teletrax_analysis.df = df

# Extract topic from slug line (text before the first slash)
df['Topic'] = df['Slug line'].apply(lambda x: x.split('-')[0] if pd.notna(x) and isinstance(x, str) and '-' in x 
                                   else x.split('/')[0] if pd.notna(x) and isinstance(x, str) and '/' in x 
                                   else x if pd.notna(x) and isinstance(x, str) 
                                   else '')
df['Subtopic'] = df['Slug line'].apply(lambda x: x.split('-')[1].split('/')[0] if pd.notna(x) and isinstance(x, str) and '-' in x and len(x.split('-')) > 1 
                                      else '')

# Get unique channels
channels = df['Channel: Name'].unique()
print(f"Found {len(channels)} channels: {', '.join(channels)}")

# Create both regular and client-friendly versions for each channel
for channel in channels:
    print(f"\nGenerating presentations for {channel}...")
    
    # Create regular version
    regular_path = create_single_slide_presentation(channel, output_dir='output', client_friendly=False)
    print(f"Regular version saved to: {regular_path}")
    
    # Create client-friendly version
    client_friendly_path = create_single_slide_presentation(channel, output_dir='output', client_friendly=True)
    print(f"Client-friendly version saved to: {client_friendly_path}")

print("\nAll presentations generated successfully!")
print("You can find the PowerPoint files in the 'output' directory.")
