import pandas as pd
import os
import matplotlib.pyplot as plt
from teletrax_analysis import create_single_slide_presentation, extract_countries_from_text, analyze_top_stories

# Create output directory if it doesn't exist
os.makedirs('test_output', exist_ok=True)

# Load the SABC data file
print("Loading SABC data file...")
df = pd.read_excel('Sample_Data/SABC 2025_08_01_09_06_57.xlsx')

# Print basic info about the data
print(f"Data loaded. {len(df)} records.")
print(f"Columns: {df.columns.tolist()}")
print(f"NaN values in Slug line: {df['Slug line'].isna().sum()}")
print(f"NaN values in Headline: {df['Headline'].isna().sum()}")

# Set the channel name
channel_name = 'SABC News'

# Filter data for the specified channel
channel_df = df[df['Channel: Name'] == channel_name]
print(f"Found {len(channel_df)} records for channel {channel_name}")

# Process datetime columns
datetime_cols = ['UTC detection start', 'Local detection start']
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])

# Add derived columns for analysis
df['Detection Year'] = df['UTC detection start'].dt.year
df['Detection Month'] = df['UTC detection start'].dt.month
df['Detection Day'] = df['UTC detection start'].dt.day
df['Detection Hour'] = df['UTC detection start'].dt.hour
df['Detection Weekday'] = df['UTC detection start'].dt.day_name()
df['Detection Date'] = df['UTC detection start'].dt.date

# Extract topic from slug line (handle NaN values)
df['Topic'] = df['Slug line'].apply(lambda x: x.split('-')[0] if pd.notna(x) and '-' in x else x.split('/')[0] if pd.notna(x) else '')
df['Subtopic'] = df['Slug line'].apply(lambda x: x.split('-')[1].split('/')[0] if pd.notna(x) and '-' in x and len(x.split('-')) > 1 else '')

# Test the extract_countries_from_text function with some sample data
print("\nTesting extract_countries_from_text function:")
test_texts = [
    "BRITAIN-EU/CELEBRATIONS",
    "Tears of joy as the clock strikes Brexit",
    None,
    float('nan'),
    "SOUTH AFRICA-POLITICS/ZUMA",
    "USA-ELECTION/BIDEN"
]

for text in test_texts:
    countries = extract_countries_from_text(text) if pd.notna(text) else []
    print(f"Text: {text}, Countries: {countries}")

# Test the Top Stories analysis with the SABC data
print("\nTesting Top Stories analysis with SABC data...")
try:
    # We need to set the global df variable in the teletrax_analysis module to our SABC data
    import teletrax_analysis
    # Save the original df
    original_df = teletrax_analysis.df
    # Set the df to our SABC data
    teletrax_analysis.df = df
    
    # Run the Top Stories analysis
    analyze_top_stories()
    print("Top Stories analysis completed successfully.")
    
    # Create a single-slide presentation for the channel
    print("\nCreating single-slide presentation...")
    output_path = create_single_slide_presentation(channel_name, output_dir='test_output')
    print(f"Presentation created successfully at {output_path}")
    
    # Restore the original df
    teletrax_analysis.df = original_df
except Exception as e:
    print(f"Error in analysis: {str(e)}")
    import traceback
    traceback.print_exc()
