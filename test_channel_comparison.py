import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import re

# Create output directory if it doesn't exist
os.makedirs('test_output', exist_ok=True)

def clean_headline(text):
    """Optimized function to clean headline text"""
    if not isinstance(text, str) or pd.isna(text):
        return "No headline"
    
    # Combine regex operations to reduce processing time
    clean_text = re.sub(r'<.*?>|\s+', ' ', text).strip()
    
    # Replace "ADVISORY " prefix with "LIVE: " if present
    if clean_text.startswith("ADVISORY "):
        clean_text = "LIVE: " + clean_text[9:].strip()
    
    # If the text is empty after cleaning, return "No headline"
    if not clean_text or clean_text.isspace():
        return "No headline"
    
    # Truncate long headlines to a reasonable length (60 characters)
    if len(clean_text) > 60:
        clean_text = clean_text[:57] + "..."
        
    return clean_text

def test_channel_comparison_by_id(file_path):
    """Test the optimized Channel Comparison Analysis by Story ID functionality"""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    # Load the data
    df = pd.read_excel(file_path)
    
    print(f"Data loaded. {len(df)} records.")
    print(f"Loading time: {time.time() - start_time:.2f} seconds")
    
    # Process datetime columns
    datetime_cols = ['UTC detection start', 'Local detection start']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Start timing the analysis
    analysis_start_time = time.time()
    
    print("Generating channel comparison by Story ID...")
    
    # Get top 5 story IDs by count first - this limits our data processing scope
    top_5_story_ids = df['Story ID'].value_counts().head(5).index.tolist()
    
    # Create a filtered DataFrame with only the top 5 story IDs
    filtered_df = df[df['Story ID'].isin(top_5_story_ids)]
    
    # Process headlines only for the top 5 story IDs
    story_id_headlines = {}
    for story_id in top_5_story_ids:
        try:
            # Get headlines for this story ID from the filtered DataFrame
            story_headlines = filtered_df[filtered_df['Story ID'] == story_id]['Headline']
            if not story_headlines.empty:
                # Get the most common headline
                headline_counts = story_headlines.value_counts()
                if not headline_counts.empty and pd.notna(headline_counts.index[0]):
                    most_common_headline = clean_headline(headline_counts.index[0])
                else:
                    most_common_headline = f"Story ID: {story_id}"
            else:
                most_common_headline = f"Story ID: {story_id}"
            
            story_id_headlines[story_id] = most_common_headline
        except Exception as e:
            print(f"Error processing headline for Story ID {story_id}: {str(e)}")
            story_id_headlines[story_id] = f"Story ID: {story_id}"
    
    # Create a crosstab of Story ID vs Channel using the filtered DataFrame
    channel_story_id_counts = pd.crosstab(filtered_df['Story ID'], filtered_df['Channel: Name'])
    
    # Ensure all top 5 story IDs are in the crosstab
    for story_id in top_5_story_ids:
        if story_id not in channel_story_id_counts.index:
            # If a story ID is missing, add it with zeros
            channel_story_id_counts.loc[story_id] = 0
    
    # Keep only the top 5 story IDs in the crosstab
    channel_story_id_counts = channel_story_id_counts.loc[top_5_story_ids]
    
    # Create a mapping dictionary from story ID to headline
    id_to_headline = {story_id: story_id_headlines.get(story_id, f"Story ID: {story_id}") 
                     for story_id in top_5_story_ids}
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each channel as a separate bar
    bottom = np.zeros(len(top_5_story_ids))
    colors = plt.cm.viridis(np.linspace(0, 1, len(channel_story_id_counts.columns)))
    
    for i, channel in enumerate(channel_story_id_counts.columns):
        values = channel_story_id_counts[channel].values
        plt.barh([id_to_headline[sid] for sid in top_5_story_ids], 
                values, left=bottom, color=colors[i], label=channel)
        bottom += values
    
    plt.title('Channel Preference for Top 5 Stories (by Story ID)', fontsize=16)
    plt.xlabel('Number of Detections', fontsize=12)
    plt.ylabel('Most Common Headline', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend(title='Channel')
    
    plt.tight_layout()
    plt.savefig(os.path.join('test_output', 'channel_story_id_preference.png'), dpi=300)
    plt.close()
    
    analysis_time = time.time() - analysis_start_time
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    
    return analysis_time

if __name__ == "__main__":
    # Test with SABC data
    sabc_file = 'Sample_Data/SABC 2025_08_01_09_06_57.xlsx'
    if os.path.exists(sabc_file):
        sabc_time = test_channel_comparison_by_id(sabc_file)
        print(f"SABC data analysis time: {sabc_time:.2f} seconds")
    else:
        print(f"SABC data file not found: {sabc_file}")
    
    # Test with TV5 data
    tv5_file = 'Sample_Data/TV5 test_2025_07_23_18_19_36.xlsx'
    if os.path.exists(tv5_file):
        tv5_time = test_channel_comparison_by_id(tv5_file)
        print(f"TV5 data analysis time: {tv5_time:.2f} seconds")
    else:
        print(f"TV5 data file not found: {tv5_file}")
