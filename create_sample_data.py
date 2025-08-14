import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

# Create Sample_Data directory if it doesn't exist
os.makedirs('Sample_Data', exist_ok=True)

# Define the number of records to generate
num_records = 1000

# Define the date range for the data
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 7, 1)
date_range = (end_date - start_date).days

# Define sample channels
channels = ['SABC', 'BBC World', 'Al Jazeera', 'CNN', 'France 24']

# Define sample markets (countries)
markets = ['South Africa', 'United Kingdom', 'Qatar', 'United States', 'France', 
           'Germany', 'Japan', 'Australia', 'Brazil', 'India', 'China', 'Russia']

# Define sample regions
regions = ['EMEA', 'Americas', 'APAC']

# Define sample topics
topics = ['POLITICS', 'HEALTH', 'CLIMATE', 'CONFLICT', 'ECONOMY', 'SPORTS', 
          'ENTERTAINMENT', 'TECHNOLOGY', 'SCIENCE', 'EDUCATION']

# Define sample subtopics
subtopics = ['ELECTION', 'PANDEMIC', 'WARMING', 'WAR', 'INFLATION', 'OLYMPICS', 
             'AWARDS', 'AI', 'SPACE', 'UNIVERSITY']

# Generate random data
data = []
for i in range(num_records):
    # Generate random date within the range
    random_days = random.randint(0, date_range)
    detection_date = start_date + timedelta(days=random_days)
    
    # Generate random time
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    detection_time = detection_date.replace(hour=random_hours, minute=random_minutes, second=random_seconds)
    
    # Generate random detection length (between 5 and 120 seconds)
    detection_length_seconds = random.randint(5, 120)
    detection_length = detection_length_seconds  # Store as seconds, not timedelta
    
    # Generate random story ID
    story_id = f"RVN{random.randint(10000, 99999)}"
    
    # Generate random topic and subtopic
    topic = random.choice(topics)
    subtopic = random.choice(subtopics)
    
    # Generate random slug line
    slug_line = f"{topic}-{subtopic}/{story_id}"
    
    # Generate random headline
    headline = f"Reuters: {topic} - {subtopic} story about {random.choice(markets)}"
    
    # Generate random channel, market, and region
    channel = random.choice(channels)
    market = random.choice(markets)
    region = random.choice(regions)
    
    # Create a record
    record = {
        'Story ID': story_id,
        'Slug line': slug_line,
        'Headline': headline,
        'Duration (secs)': random.randint(30, 300),
        'Channel: Name': channel,
        'Market: Name': market,
        'Region: Name': region,
        'UTC detection start': detection_time,
        'Local detection start': detection_time + timedelta(hours=random.randint(-12, 12)),
        'Detection duration': detection_length_seconds,
        'Actual detection length': timedelta(seconds=detection_length_seconds),  # Store as timedelta
        'Hit: Activation date start (UTC)': detection_time - timedelta(days=random.randint(1, 7)),
        'Asset: Activation date start (UTC)': detection_time - timedelta(days=random.randint(1, 7))
    }
    
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
output_file = 'Sample_Data/Sample_Teletrax_Data.xlsx'
df.to_excel(output_file, index=False)

print(f"Sample data created and saved to {output_file}")
print(f"Generated {num_records} records spanning from {start_date.date()} to {end_date.date()}")
