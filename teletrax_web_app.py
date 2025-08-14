import os
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import shutil
import uuid
import threading
import markdown
import re
import numpy as np
import time
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, make_response
from werkzeug.utils import secure_filename
from litellm_client import LiteLLMClient

# Load environment variables from .env file
import load_env

# Import the analysis functions from teletrax_analysis.py
import teletrax_analysis
from teletrax_analysis import (
    generate_time_series_analysis,
    analyze_top_stories,
    analyze_detection_patterns,
    analyze_detection_lengths,
    timedelta_to_seconds
)

def convert_markdown_to_html(text):
    """Convert markdown text to HTML with proper formatting
    
    This function takes markdown text and converts it to properly formatted HTML.
    It handles headings, lists, bold/italic text, and other markdown elements.
    
    Args:
        text (str): Markdown text to convert
        
    Returns:
        str: HTML formatted text
    """
    if not text:
        return ""
    
    # Convert markdown to HTML
    html = markdown.markdown(text)
    
    # Improve formatting for lists
    # Add proper indentation and spacing for lists
    html = re.sub(r'<ul>', r'<ul class="ai-list">', html)
    html = re.sub(r'<ol>', r'<ol class="ai-list">', html)
    
    # Improve formatting for headings
    html = re.sub(r'<h3>', r'<h3 class="ai-heading">', html)
    html = re.sub(r'<h4>', r'<h4 class="ai-heading">', html)
    
    # Add spacing between paragraphs
    html = re.sub(r'<p>', r'<p class="ai-paragraph">', html)
    
    return html

app = Flask(__name__)
app.secret_key = 'teletrax_analysis_key'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('file')
    
    if not files or all(file.filename == '' for file in files):
        flash('No selected files')
        return redirect(request.url)
    
    # Check if all files have valid extensions
    if not all(allowed_file(file.filename) for file in files):
        flash('Invalid file type. Please upload only Excel or CSV files.')
        return redirect(url_for('index'))
    
    # Create a unique session ID
    session_id = str(uuid.uuid4())
    
    # Create session directory
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    # Create output directories for this session
    output_dir = os.path.join('static', 'images', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files
    try:
        # List to store DataFrames from each file
        dfs = []
        file_info = []
        
        # Process each file
        for file in files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_dir, filename)
            file.save(file_path)
            
            # Load the data
            if filename.endswith('.csv'):
                file_df = pd.read_csv(file_path)
            else:
                file_df = pd.read_excel(file_path)
            
            # Check if the file has the expected columns
            required_columns = [
                'Channel: Name', 'Market: Name', 'UTC detection start', 
                'Local detection start', 'Story ID', 'Slug line', 
                'Actual detection length'
            ]
            
            missing_columns = [col for col in required_columns if col not in file_df.columns]
            if missing_columns:
                file_info.append({
                    'filename': filename,
                    'status': 'error',
                    'message': f"Missing required columns: {', '.join(missing_columns)}"
                })
                continue
            
            # Add source file information
            file_df['Source File'] = filename
            
            # Add to the list of DataFrames
            dfs.append(file_df)
            file_info.append({
                'filename': filename,
                'status': 'success',
                'records': len(file_df),
                'channels': ', '.join(file_df['Channel: Name'].unique())
            })
        
        # Check if any files were successfully processed
        if not dfs:
            flash("None of the uploaded files could be processed. Please check the file format.")
            return redirect(url_for('index'))
        
        # Combine all DataFrames
        df = pd.concat(dfs, ignore_index=True)
        
        # Process datetime columns
        datetime_cols = ['UTC detection start', 'Local detection start']
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert 'Actual detection length' to seconds
        if 'Actual detection length' in df.columns:
            df['Detection Length (seconds)'] = df['Actual detection length'].apply(timedelta_to_seconds)
        
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
        
        # Save processed data
        processed_file = os.path.join(session_dir, 'processed_data.pkl')
        df.to_pickle(processed_file)
        
        # Save file processing information
        file_info_path = os.path.join(session_dir, 'file_info.json')
        with open(file_info_path, 'w') as f:
            json.dump(file_info, f)
        
        # Display success message with file information
        if len(file_info) > 1:
            successful_files = [info['filename'] for info in file_info if info['status'] == 'success']
            flash(f"Successfully processed {len(successful_files)} files with a total of {len(df)} records.")
        
        # Redirect to analysis page
        return redirect(url_for('analyze', session_id=session_id))
        
    except Exception as e:
        flash(f"Error processing files: {str(e)}")
        return redirect(url_for('index'))

@app.route('/analyze/<session_id>')
def analyze(session_id):
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load processed data
    processed_file = os.path.join(session_dir, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        flash('Processed data not found')
        return redirect(url_for('index'))
    
    df = pd.read_pickle(processed_file)
    
    # Get basic stats for display
    stats = {
        'total_records': len(df),
        'date_range': f"{df['UTC detection start'].min().date()} to {df['UTC detection start'].max().date()}",
        'channels': ', '.join(df['Channel: Name'].unique()),
        'markets': ', '.join(df['Market: Name'].unique()),
        'unique_stories': df['Slug line'].nunique(),
        'unique_story_ids': df['Story ID'].nunique()
    }
    
    # Get top 5 stories for display
    # We'll still use slug lines for the dashboard overview
    top_stories = df['Slug line'].value_counts().head(5).to_dict()
    
    # Load file processing information if available
    file_info = []
    file_info_path = os.path.join(session_dir, 'file_info.json')
    if os.path.exists(file_info_path):
        try:
            with open(file_info_path, 'r') as f:
                file_info = json.load(f)
        except Exception as e:
            print(f"Error loading file info: {str(e)}")
    
    return render_template('analyze.html', 
                          session_id=session_id, 
                          stats=stats, 
                          top_stories=top_stories,
                          file_info=file_info)

@app.route('/generate/<analysis_type>/<session_id>')
def generate_analysis(analysis_type, session_id):
    """Generate a specific analysis visualization
    
    This function generates a specific analysis visualization based on the
    analysis_type parameter. It supports various analysis types including
    top_stories which now includes both Story ID + Headline and Slug Line
    organization options, as well as story_distribution which shows the
    geographic distribution of stories.
    
    Args:
        analysis_type (str): Type of analysis to generate
        session_id (str): Unique session ID for the current analysis
        
    Returns:
        Redirect to the view_analysis page with the generated image
    """
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load processed data
    processed_file = os.path.join(session_dir, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        flash('Processed data not found')
        return redirect(url_for('index'))
    
    df = pd.read_pickle(processed_file)
    
    # Set output directory for this session
    output_dir = os.path.join('static', 'images', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Temporarily redirect matplotlib output
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
    # Generate the requested analysis
    if analysis_type == 'time_series':
        # Time series analysis
        daily_counts = df.groupby('Detection Date').size()
        date_range = pd.date_range(start=daily_counts.index.min(), end=daily_counts.index.max())
        daily_counts = daily_counts.reindex(date_range, fill_value=0)
        moving_avg = daily_counts.rolling(window=30).mean()
        
        plt.figure(figsize=(15, 7))
        plt.plot(daily_counts.index, daily_counts.values, 'b-', alpha=0.5, label='Daily Detections')
        plt.plot(moving_avg.index, moving_avg.values, 'r-', linewidth=2, label='30-Day Moving Average')
        plt.title('Reuters Video Detections Over Time', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        import matplotlib.dates as mdates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series_analysis.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='time_series_analysis.png'))
    
    elif analysis_type == 'top_stories':
        # Group by Story ID and find the most common headline for each ID
        story_id_counts = df.groupby('Story ID').size().sort_values(ascending=False).head(15)
        
        # Function to clean headline text
        def clean_headline(text):
            if not isinstance(text, str):
                return "No headline"
            
            # Remove HTML tags
            import re
            clean_text = re.sub(r'<.*?>', '', text)
            
            # Replace "ADVISORY " prefix with "LIVE: " if present (ADVISORY indicates a live broadcast)
            if clean_text.startswith("ADVISORY "):
                clean_text = "LIVE: " + clean_text[9:].strip()
            
            # Remove special characters and normalize whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # If the text is empty or just whitespace after cleaning, return "No headline"
            if not clean_text or clean_text.isspace():
                return "No headline"
            
            # Truncate long headlines to a reasonable length (60 characters)
            if len(clean_text) > 60:
                clean_text = clean_text[:57] + "..."
                
            return clean_text
        
        # For each Story ID, find the most common headline
        story_id_headlines = {}
        for story_id in story_id_counts.index:
            headlines = df[df['Story ID'] == story_id]['Headline'].value_counts()
            if not headlines.empty and pd.notna(headlines.index[0]):
                most_common_headline = clean_headline(headlines.index[0])
            else:
                most_common_headline = "No headline"
            story_id_headlines[story_id] = most_common_headline
        
        # Create labels with only the most common headline (no Story ID)
        labels = [f"{story_id_headlines[story_id]}" for story_id in story_id_counts.index]
        
        plt.figure(figsize=(14, 10))
        bars = plt.barh(labels[::-1], story_id_counts.values[::-1], color='skyblue')
        plt.title('Top 15 Most Detected Stories (by Story ID)', fontsize=16)
        plt.xlabel('Number of Detections', fontsize=12)
        plt.ylabel('Most Common Headline', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Set a fixed x-axis limit to prevent stretching
        # Calculate a reasonable maximum based on the data
        max_value = story_id_counts.max()
        plt.xlim(0, max_value * 1.2)  # Add 20% padding
        
        # Add explanatory note
        plt.figtext(0.5, 0.01, 
                   "Note: Stories are grouped by Story ID. Each story is displayed with its most commonly used headline.", 
                   ha='center', fontsize=10, style='italic')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{width:,.0f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_stories_by_id.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='top_stories_by_id.png'))
    
    elif analysis_type == 'top_themes':
        # Extract master slugs (portion before the forward slash)
        def extract_master_slug(slug_line):
            if pd.isna(slug_line):
                return "Unknown"
            
            # Replace "ADVISORY " prefix with "LIVE: " if present (ADVISORY indicates a live broadcast)
            if isinstance(slug_line, str) and slug_line.startswith("ADVISORY "):
                slug_line = "LIVE: " + slug_line[9:].strip()
            
            # Find the position of the first forward slash
            slash_pos = slug_line.find('/')
            
            # If a slash is found, extract everything before it and add the slash
            if slash_pos != -1:
                return slug_line[:slash_pos+1]  # Include the slash
            else:
                # If no slash is found, return the entire slug line
                return slug_line
        
        # Apply the function to extract master slugs
        df['Master Slug'] = df['Slug line'].apply(extract_master_slug)
        
        # Count occurrences of each master slug
        top_master_slugs = df['Master Slug'].value_counts().head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_master_slugs.index[::-1], top_master_slugs.values[::-1], color='skyblue')
        plt.title('Top 15 Thematic Areas (by Master Slug)', fontsize=16)
        plt.xlabel('Number of Detections', fontsize=12)
        plt.ylabel('Master Slug', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add count labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                    f'{width:,.0f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_themes.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type='top_themes',
                               image='top_themes.png'))
    
    elif analysis_type == 'detection_patterns':
        # Hour of day analysis
        hourly_counts = df['Detection Hour'].value_counts().sort_index()
        
        plt.figure(figsize=(12, 6))
        plt.bar(hourly_counts.index, hourly_counts.values, color='purple', alpha=0.7)
        plt.title('Detections by Hour of Day (UTC)', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Number of Detections', fontsize=12)
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hourly_detection_pattern.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='hourly_detection_pattern.png'))
    
    elif analysis_type == 'detection_lengths':
        # Detection length distribution
        plt.figure(figsize=(12, 6))
        plt.hist(df['Detection Length (seconds)'], bins=30, color='orange', alpha=0.7)
        plt.title('Distribution of Detection Lengths', fontsize=16)
        plt.xlabel('Detection Length (seconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_length_distribution.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='detection_length_distribution.png'))
    
    elif analysis_type == 'channel_comparison':
        # Channel comparison by Slug line (original implementation)
        channel_story_counts = pd.crosstab(df['Slug line'], df['Channel: Name'])
        top_5_stories = df['Slug line'].value_counts().head(5).index
        channel_story_counts = channel_story_counts.loc[top_5_stories]
        
        plt.figure(figsize=(12, 8))
        channel_story_counts.plot(kind='barh', stacked=True, figsize=(12, 8), 
                                 colormap='viridis')
        plt.title('Channel Preference for Top 5 Stories (by Slug Line)', fontsize=16)
        plt.xlabel('Number of Detections', fontsize=12)
        plt.ylabel('Story Slug Line', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend(title='Channel')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'channel_story_preference.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='channel_story_preference.png'))
                               
    elif analysis_type == 'channel_comparison_by_id':
        # Channel comparison by Story ID - Optimized for performance
        try:
            print("Generating channel comparison by Story ID...")
            
            # Get top 5 story IDs by count first - this limits our data processing scope
            top_5_story_ids = df['Story ID'].value_counts().head(5).index.tolist()
            
            # Create a filtered DataFrame with only the top 5 story IDs
            filtered_df = df[df['Story ID'].isin(top_5_story_ids)]
            
            # Optimized function to clean headline text
            def clean_headline(text):
                if not isinstance(text, str) or pd.isna(text):
                    return "No headline"
                
                # Combine regex operations to reduce processing time
                import re
                # Remove HTML tags and normalize whitespace in one pass
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
            
            # Create a copy of the DataFrame with headline labels
            plot_df = channel_story_id_counts.copy()
            
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
            plt.savefig(os.path.join(output_dir, 'channel_story_id_preference.png'), dpi=300)
            plt.close()
            
            return redirect(url_for('view_analysis', 
                                   session_id=session_id, 
                                   analysis_type=analysis_type,
                                   image='channel_story_id_preference.png'))
        except Exception as e:
            print(f"Error generating channel comparison by Story ID: {str(e)}")
            flash(f"Error generating channel comparison by Story ID: {str(e)}")
            return redirect(url_for('analyze', session_id=session_id))
                               
    elif analysis_type == 'channel_comparison_by_master_slug':
        # Channel comparison by Master Slug
        
        # Extract master slugs (portion before the forward slash)
        def extract_master_slug(slug_line):
            if pd.isna(slug_line):
                return "Unknown"
            
            # Replace "ADVISORY " prefix with "LIVE: " if present
            if isinstance(slug_line, str) and slug_line.startswith("ADVISORY "):
                slug_line = "LIVE: " + slug_line[9:].strip()
            
            # Find the position of the first forward slash
            slash_pos = slug_line.find('/')
            
            # If a slash is found, extract everything before it and add the slash
            if slash_pos != -1:
                return slug_line[:slash_pos+1]  # Include the slash
            else:
                # If no slash is found, return the entire slug line
                return slug_line
        
        # Apply the function to extract master slugs
        df['Master Slug'] = df['Slug line'].apply(extract_master_slug)
        
        # Get top 5 master slugs by count
        top_5_master_slugs = df['Master Slug'].value_counts().head(5).index
        
        # Create a crosstab of Master Slug vs Channel
        channel_master_slug_counts = pd.crosstab(df['Master Slug'], df['Channel: Name'])
        channel_master_slug_counts = channel_master_slug_counts.loc[top_5_master_slugs]
        
        plt.figure(figsize=(12, 8))
        channel_master_slug_counts.plot(kind='barh', stacked=True, figsize=(12, 8), 
                                       colormap='viridis')
        plt.title('Channel Preference for Top 5 Thematic Areas (by Master Slug)', fontsize=16)
        plt.xlabel('Number of Detections', fontsize=12)
        plt.ylabel('Master Slug', fontsize=12)
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend(title='Channel')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'channel_master_slug_preference.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='channel_master_slug_preference.png'))
                               
    elif analysis_type == 'story_distribution':
        # Geographic distribution of stories
        # Extract countries from slug lines and headlines
        from teletrax_analysis import extract_countries_from_text
        
        content_countries = []
        
        # Process each story in the data
        for _, row in df.iterrows():
            # Extract countries from slug line (handle NaN values)
            slug_countries = extract_countries_from_text(row['Slug line']) if pd.notna(row['Slug line']) else []
            
            # Extract countries from headline (handle NaN values)
            headline_countries = extract_countries_from_text(row['Headline']) if pd.notna(row['Headline']) else []
            
            # Combine countries from both sources
            story_countries = list(set(slug_countries + headline_countries))
            
            # If no countries found, try using the topic as a potential country
            if not story_countries and 'Topic' in row and pd.notna(row['Topic']):
                topic_countries = extract_countries_from_text(row['Topic'])
                if topic_countries:
                    story_countries = topic_countries
            
            # Add to the overall list
            content_countries.extend(story_countries)
        
        # Count occurrences of each country
        if content_countries:
            country_counts = pd.Series(content_countries).value_counts()
            
            # If there are too many countries, group smaller ones as "Rest of the world"
            if len(country_counts) > 8:
                top_countries = country_counts.head(7)
                rest_of_world = pd.Series({'Rest of the world': country_counts[7:].sum()})
                country_counts = pd.concat([top_countries, rest_of_world])
            
            # Calculate percentages
            country_percentages = (country_counts / country_counts.sum() * 100).round(0).astype(int)
        else:
            # If no countries found, create a default "Unknown" category
            country_counts = pd.Series({'Unknown': 1})
            country_percentages = pd.Series({'Unknown': 100})
        
        # Create the pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(country_percentages, 
                labels=[f"{c}\n{p}%" for c, p in zip(country_percentages.index, country_percentages.values)], 
                autopct='', 
                startangle=90, 
                wedgeprops={'edgecolor': 'white'},
                colors=teletrax_analysis.TR_PIE_COLORS)
        plt.title("Reuters Video Content Origin by Country", fontsize=16)
        plt.axis('equal')
        
        # Save the chart
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'country_distribution_pie.png'), dpi=300)
        plt.close()
        
        return redirect(url_for('view_analysis', 
                               session_id=session_id, 
                               analysis_type=analysis_type,
                               image='country_distribution_pie.png'))
    
    else:
        flash('Invalid analysis type')
        return redirect(url_for('analyze', session_id=session_id))

@app.route('/view/<session_id>/<analysis_type>/<image>')
def view_analysis(session_id, analysis_type, image):
    image_path = f'images/{session_id}/{image}'
    
    analysis_titles = {
        'time_series': 'Time Series Analysis',
        'top_stories': 'Top Stories Analysis',
        'top_themes': 'Top Themes Analysis',
        'detection_patterns': 'Detection Patterns Analysis',
        'detection_lengths': 'Detection Lengths Analysis',
        'channel_comparison': 'Channel Comparison Analysis (by Slug Line)',
        'channel_comparison_by_id': 'Channel Comparison Analysis (by Story ID)',
        'channel_comparison_by_master_slug': 'Channel Comparison Analysis (by Master Slug)',
        'story_distribution': 'Story Distribution by Country'
    }
    
    return render_template('view_analysis.html', 
                          session_id=session_id,
                          analysis_type=analysis_type,
                          analysis_title=analysis_titles.get(analysis_type, 'Analysis'),
                          image_path=image_path)

@app.route('/generate_all/<session_id>')
def generate_all_analyses(session_id):
    """Generate all analyses and create a PowerPoint presentation
    
    This function runs all the analysis functions and creates a comprehensive
    PowerPoint presentation. It handles the output directory patching to avoid
    recursion errors and ensures the PowerPoint file is properly copied to the
    static/reports directory for download.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        
    Returns:
        Redirect to the analysis page
    """
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load processed data
    processed_file = os.path.join(session_dir, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        flash('Processed data not found')
        return redirect(url_for('index'))
    
    # Set output directory for this session
    output_dir = os.path.join('static', 'images', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a temporary output directory for the analysis script
    temp_output_dir = os.path.join(session_dir, 'temp_output')
    os.makedirs(temp_output_dir, exist_ok=True)
    os.makedirs(os.path.join(temp_output_dir, 'images'), exist_ok=True)
    
    try:
        # Load the data
        df = pd.read_pickle(processed_file)
        
        # Save the original output directory
        original_output_dir = 'output'
        
        # Temporarily redirect output to the session directory
        import teletrax_analysis
        # Store the original makedirs function to avoid recursion
        original_makedirs = os.makedirs
        teletrax_analysis.os.makedirs = lambda path, exist_ok=True: original_makedirs(path.replace(original_output_dir, temp_output_dir), exist_ok=True)
        
        # Store the original df from teletrax_analysis
        original_df = teletrax_analysis.df
        
        # Set the df to our session data
        teletrax_analysis.df = df
        
        # Run all analyses
        teletrax_analysis.main()
        
        # Restore the original df
        teletrax_analysis.df = original_df
        
        # Copy all generated files to the static directory
        for root, dirs, files in os.walk(os.path.join(temp_output_dir, 'images')):
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(output_dir, file)
                shutil.copy2(src_file, dst_file)
        
        # Copy PowerPoint if generated
        # Check both possible locations for the PowerPoint file
        ppt_file = os.path.join(temp_output_dir, 'RVN_Client_Analysis.pptx')
        if not os.path.exists(ppt_file):
            # Try the original output directory
            ppt_file = os.path.join('output', 'RVN_Client_Analysis.pptx')
        
        if os.path.exists(ppt_file):
            dst_ppt = os.path.join('static', 'reports', f'{session_id}_RVN_Client_Analysis.pptx')
            shutil.copy2(ppt_file, dst_ppt)
            print(f"PowerPoint copied from {ppt_file} to {dst_ppt}")
        else:
            print(f"PowerPoint file not found in {temp_output_dir} or output/")
            
            # As a fallback, run the PowerPoint generation directly
            try:
                # Import the create_powerpoint_presentation function
                from teletrax_analysis import create_powerpoint_presentation
                
                # Run the function to create a new PowerPoint with the correct output directory
                create_powerpoint_presentation(output_dir='output')
                
                # Now try to copy from the output directory again
                ppt_file = os.path.join('output', 'RVN_Client_Analysis.pptx')
                if os.path.exists(ppt_file):
                    dst_ppt = os.path.join('static', 'reports', f'{session_id}_RVN_Client_Analysis.pptx')
                    shutil.copy2(ppt_file, dst_ppt)
                    print(f"PowerPoint copied from {ppt_file} to {dst_ppt} after direct generation")
                else:
                    print("Failed to generate PowerPoint file")
            except Exception as e:
                print(f"Error generating PowerPoint directly: {str(e)}")
        
        flash('All analyses generated successfully!')
        return redirect(url_for('analyze', session_id=session_id))
        
    except Exception as e:
        flash(f"Error generating analyses: {str(e)}")
        return redirect(url_for('analyze', session_id=session_id))
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_output_dir, ignore_errors=True)

@app.route('/download_report/<session_id>')
def download_report(session_id):
    """Download the PowerPoint presentation for the current session
    
    This function checks for the PowerPoint file in the static/reports directory.
    If not found, it attempts to regenerate it using the session data. If still not
    found, it displays an error message.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        
    Returns:
        The PowerPoint file for download or a redirect to the analysis page
    """
    # Check for the report in the static/reports directory
    report_file = os.path.join('static', 'reports', f'{session_id}_RVN_Client_Analysis.pptx')
    
    if not os.path.exists(report_file):
        # If not found, try to regenerate it
        try:
            # Check if session exists
            session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            if not os.path.exists(session_dir):
                flash('Session not found')
                return redirect(url_for('index'))
            
            # Load processed data
            processed_file = os.path.join(session_dir, 'processed_data.pkl')
            if not os.path.exists(processed_file):
                flash('Processed data not found')
                return redirect(url_for('index'))
            
            # Load the data
            df = pd.read_pickle(processed_file)
            
            # Import the create_powerpoint_presentation function
            import teletrax_analysis
            
            # Store the original df
            original_df = teletrax_analysis.df
            
            # Set the df to our session data
            teletrax_analysis.df = df
            
            # Run the function to create a new PowerPoint
            teletrax_analysis.create_powerpoint_presentation(output_dir='output')
            
            # Restore the original df
            teletrax_analysis.df = original_df
            
            # Now try to copy from the output directory
            output_ppt = os.path.join('output', 'RVN_Client_Analysis.pptx')
            if os.path.exists(output_ppt):
                os.makedirs(os.path.join('static', 'reports'), exist_ok=True)
                shutil.copy2(output_ppt, report_file)
                print(f"PowerPoint regenerated and copied from {output_ppt} to {report_file}")
            else:
                print("Failed to generate PowerPoint file")
                flash('Failed to generate PowerPoint file. Please try generating all analyses first.')
                return redirect(url_for('analyze', session_id=session_id))
        except Exception as e:
            print(f"Error regenerating PowerPoint: {str(e)}")
            flash(f"Error generating PowerPoint: {str(e)}")
            return redirect(url_for('analyze', session_id=session_id))
    
    if os.path.exists(report_file):
        return send_from_directory('static/reports', f'{session_id}_RVN_Client_Analysis.pptx', as_attachment=True)
    else:
        flash('Report not found. Please generate all analyses first.')
        return redirect(url_for('analyze', session_id=session_id))

@app.route('/generate_single_slide/<session_id>/<channel_name>')
def generate_single_slide(session_id, channel_name):
    """Generate a single-slide PowerPoint presentation for a specific channel
    
    This function creates a simple single-slide PowerPoint presentation for a specific
    channel, following the format of the sample slides. The slide has a 2x2 grid layout
    with time series chart, pie chart, text summary, and major stories.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        channel_name (str): Name of the channel to create the presentation for
        
    Returns:
        Redirect to the analysis page
    """
    # Check if client_friendly parameter is set
    client_friendly = request.args.get('client_friendly', '0') == '1'
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load processed data
    processed_file = os.path.join(session_dir, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        flash('Processed data not found')
        return redirect(url_for('index'))
    
    # Set output directory for this session
    output_dir = os.path.join('static', 'images', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the data
        df = pd.read_pickle(processed_file)
        
        # Import the create_single_slide_presentation function
        from teletrax_analysis import create_single_slide_presentation
        
        # Store the original df from teletrax_analysis
        import teletrax_analysis
        original_df = teletrax_analysis.df
        
        # Set the df to our session data
        teletrax_analysis.df = df
        
        # Create the single-slide presentation
        output_path = create_single_slide_presentation(channel_name, output_dir=output_dir, client_friendly=client_friendly)
        
        # Restore the original df
        teletrax_analysis.df = original_df
        
        # Copy the presentation to the static/reports directory
        if output_path and os.path.exists(output_path):
            dst_path = os.path.join('static', 'reports', f'{session_id}_{channel_name}_Single_Slide.pptx')
            shutil.copy2(output_path, dst_path)
            print(f"Single-slide PowerPoint copied to {dst_path}")
            
            flash(f'Single-slide PowerPoint for {channel_name} generated successfully!')
        else:
            flash(f'Failed to generate single-slide PowerPoint for {channel_name}')
        
        return redirect(url_for('analyze', session_id=session_id))
        
    except Exception as e:
        flash(f"Error generating single-slide PowerPoint: {str(e)}")
        return redirect(url_for('analyze', session_id=session_id))

@app.route('/download_single_slide/<session_id>/<channel_name>')
def download_single_slide(session_id, channel_name):
    """Download the single-slide PowerPoint presentation for a specific channel
    
    This function provides a direct download link for channel-specific single-slide
    PowerPoint presentations. It checks for the file in the static/reports directory.
    If not found, it attempts to locate it in the output directory and copies it.
    If still not found, it redirects to the generate_single_slide route to create it.
    
    The single-slide PowerPoint contains:
    1. A time series chart showing video usage over time with a trend line
    2. A pie chart showing video usage by country
    3. A text summary with key statistics and trends
    4. Major stories/topics with percentages
    
    This feature is particularly useful for:
    - Quick presentations to editorial teams
    - Channel-specific reporting
    - Executive summaries
    - Comparing usage patterns between different channels
    
    Args:
        session_id (str): Unique session ID for the current analysis
        channel_name (str): Name of the channel to download the presentation for
        
    Returns:
        The PowerPoint file for download or a redirect to generate the file if not found
    """
    # Check if client_friendly parameter is set
    client_friendly = request.args.get('client_friendly', '0') == '1'
    # Check for the report in the static/reports directory
    if client_friendly:
        report_file = os.path.join('static', 'reports', f'{session_id}_{channel_name}_Client_Friendly_Single_Slide.pptx')
    else:
        report_file = os.path.join('static', 'reports', f'{session_id}_{channel_name}_Single_Slide.pptx')
    
    if not os.path.exists(report_file):
        # If not found, check if the PowerPoint exists in the output directory
        output_dir = os.path.join('static', 'images', session_id)
        if client_friendly:
            output_ppt = os.path.join(output_dir, f'{channel_name}_Client_Friendly_Single_Slide.pptx')
        else:
            output_ppt = os.path.join(output_dir, f'{channel_name}_Single_Slide.pptx')
        
        if os.path.exists(output_ppt):
            # Copy it to the static/reports directory
            os.makedirs(os.path.join('static', 'reports'), exist_ok=True)
            shutil.copy2(output_ppt, report_file)
            print(f"Single-slide PowerPoint copied from {output_ppt} to {report_file}")
    
    if os.path.exists(report_file):
        filename = os.path.basename(report_file)
        return send_from_directory('static/reports', filename, as_attachment=True)
    else:
        # Try to generate it
        return redirect(url_for('generate_single_slide', session_id=session_id, channel_name=channel_name, client_friendly=int(client_friendly)))

@app.route('/generate_ai_analysis/<session_id>/<channel_name>')
def generate_ai_analysis(session_id, channel_name):
    """Generate AI-powered analysis for a specific channel
    
    This function uses the LiteLLM API to generate AI-powered insights and recommendations
    based on the Teletrax data for a specific channel. It processes the data, sends it to
    the LiteLLM API, and stores the results for display.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        channel_name (str): Name of the channel to analyze
        
    Returns:
        Redirect to the view_ai_analysis page with the generated insights
    """
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Load processed data
    processed_file = os.path.join(session_dir, 'processed_data.pkl')
    if not os.path.exists(processed_file):
        flash('Processed data not found')
        return redirect(url_for('index'))
    
    # Check if regenerate parameter is set
    regenerate = request.args.get('regenerate', '0') == '1'
    
    # Check if analysis already exists and regenerate is not set
    analysis_file = os.path.join(session_dir, f'ai_analysis_{channel_name}.json')
    if os.path.exists(analysis_file) and not regenerate:
        return redirect(url_for('view_ai_analysis', session_id=session_id, channel_name=channel_name))
    
    # Set output directory for this session
    output_dir = os.path.join('static', 'images', session_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a flag file to indicate analysis is in progress
    with open(os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag'), 'w') as f:
        f.write('1')
    
    # Start the analysis in a background thread
    def run_analysis():
        try:
            # Load the data
            df = pd.read_pickle(processed_file)
            
            # Filter data for the specific channel
            channel_df = df[df['Channel: Name'] == channel_name]
            
            if len(channel_df) == 0:
                raise ValueError(f"No data found for channel {channel_name}")
            
            # Get basic stats for the channel
            stats = {
                'total_records': len(channel_df),
                'date_range': f"{channel_df['UTC detection start'].min().date()} to {channel_df['UTC detection start'].max().date()}",
                'markets': ', '.join(channel_df['Market: Name'].unique()),
                'unique_stories': channel_df['Slug line'].nunique(),
                'unique_story_ids': channel_df['Story ID'].nunique()
            }
            
            # Get top stories for the channel
            top_stories = channel_df['Slug line'].value_counts().head(10).to_dict()
            
            # Extract top themes (master slugs)
            def extract_master_slug(slug_line):
                if pd.isna(slug_line):
                    return "Unknown"
                
                # Replace "ADVISORY " prefix with "LIVE: " if present (ADVISORY indicates a live broadcast)
                if isinstance(slug_line, str) and slug_line.startswith("ADVISORY "):
                    slug_line = "LIVE: " + slug_line[9:].strip()
                
                # Find the position of the first forward slash
                slash_pos = slug_line.find('/')
                
                # If a slash is found, extract everything before it and add the slash
                if slash_pos != -1:
                    return slug_line[:slash_pos+1]  # Include the slash
                else:
                    # If no slash is found, return the entire slug line
                    return slug_line
            
            channel_df['Master Slug'] = channel_df['Slug line'].apply(extract_master_slug)
            top_themes = channel_df['Master Slug'].value_counts().head(10).to_dict()
            
            # Get detection patterns
            hourly_counts = channel_df['Detection Hour'].value_counts().sort_index()
            peak_hours = hourly_counts.nlargest(3).index.tolist()
            
            weekday_counts = channel_df['Detection Weekday'].value_counts()
            peak_days = weekday_counts.nlargest(3).index.tolist()
            
            detection_patterns = {
                'peak_hours': ', '.join([f"{hour}:00" for hour in peak_hours]),
                'peak_days': ', '.join(peak_days)
            }
            
            # Get country distribution
            from teletrax_analysis import extract_countries_from_text
            
            content_countries = []
            
            # Process each story in the data
            for _, row in channel_df.iterrows():
                # Extract countries from slug line (handle NaN values)
                slug_countries = extract_countries_from_text(row['Slug line']) if pd.notna(row['Slug line']) else []
                
                # Extract countries from headline (handle NaN values)
                headline_countries = extract_countries_from_text(row['Headline']) if pd.notna(row['Headline']) else []
                
                # Combine countries from both sources
                story_countries = list(set(slug_countries + headline_countries))
                
                # If no countries found, try using the topic as a potential country
                if not story_countries and 'Topic' in row and pd.notna(row['Topic']):
                    topic_countries = extract_countries_from_text(row['Topic'])
                    if topic_countries:
                        story_countries = topic_countries
                
                # Add to the overall list
                content_countries.extend(story_countries)
            
            # Count occurrences of each country
            if content_countries:
                country_counts = pd.Series(content_countries).value_counts()
                
                # If there are too many countries, group smaller ones as "Rest of the world"
                if len(country_counts) > 8:
                    top_countries = country_counts.head(7)
                    rest_of_world = pd.Series({'Rest of the world': country_counts[7:].sum()})
                    country_counts = pd.concat([top_countries, rest_of_world])
                
                # Calculate percentages
                country_percentages = (country_counts / country_counts.sum() * 100).round(0).astype(int)
                country_distribution = country_percentages.to_dict()
            else:
                # If no countries found, create a default "Unknown" category
                country_distribution = {'Unknown': 100}
            
            # Prepare the data for the LiteLLM API
            teletrax_data = {
                'stats': stats,
                'top_stories': top_stories,
                'top_themes': top_themes,
                'detection_patterns': detection_patterns,
                'country_distribution': country_distribution
            }
            
            # Initialize the LiteLLM client
            # Use the API key from environment variable
            api_key = os.environ.get('LITELLM_API_KEY')
            api_url = os.environ.get('LITELLM_API_URL', 'https://litellm.int.thomsonreuters.com')
            
            if not api_key:
                raise ValueError("LITELLM_API_KEY environment variable is not set. Please set it in your .env file.")
            
            litellm_client = LiteLLMClient(api_key=api_key, api_url=api_url)
            
            # Generate the analysis
            analysis = litellm_client.generate_analysis(teletrax_data, channel_name)
            
            # Save the analysis to a file
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            # Remove the in-progress flag
            if os.path.exists(os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag')):
                os.remove(os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag'))
            
        except Exception as e:
            print(f"Error generating AI analysis: {str(e)}")
            
            # Create an error analysis
            error_analysis = {
                'channel_name': channel_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'executive_summary': f"Error generating analysis: {str(e)}",
                'audience_insights': {
                    'journalists_producers': "Error generating insights.",
                    'output_editors': "Error generating insights.",
                    'marketing_teams': "Error generating insights."
                },
                'insight_types': {
                    'content_strategy': "Error generating insights.",
                    'client_engagement': "Error generating insights.",
                    'market_positioning': "Error generating insights."
                },
                'recommendations': "Error generating recommendations."
            }
            
            # Save the error analysis to a file
            with open(analysis_file, 'w') as f:
                json.dump(error_analysis, f, indent=2)
            
            # Remove the in-progress flag
            if os.path.exists(os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag')):
                os.remove(os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag'))
    
    # Start the analysis in a background thread
    analysis_thread = threading.Thread(target=run_analysis)
    analysis_thread.daemon = True
    analysis_thread.start()
    
    # Redirect to the view page, which will show a loading indicator
    return redirect(url_for('view_ai_analysis', session_id=session_id, channel_name=channel_name))

@app.route('/view_ai_analysis/<session_id>/<channel_name>')
def view_ai_analysis(session_id, channel_name):
    """View the AI-powered analysis for a specific channel
    
    This function displays the AI-generated insights and recommendations for a specific
    channel. If the analysis is still in progress, it shows a loading indicator.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        channel_name (str): Name of the channel being analyzed
        
    Returns:
        Rendered template with the AI analysis
    """
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Check if analysis is in progress
    in_progress_flag = os.path.join(session_dir, f'ai_analysis_{channel_name}_in_progress.flag')
    if os.path.exists(in_progress_flag):
        # Show loading page
        return render_template('ai_analysis.html',
                              session_id=session_id,
                              analysis={'channel_name': channel_name},
                              loading=True)
    
    # Check if analysis exists
    analysis_file = os.path.join(session_dir, f'ai_analysis_{channel_name}.json')
    if not os.path.exists(analysis_file):
        # Start the analysis
        return redirect(url_for('generate_ai_analysis', session_id=session_id, channel_name=channel_name))
    
    # Load the analysis
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Convert markdown to HTML for better formatting
    analysis['executive_summary'] = convert_markdown_to_html(analysis['executive_summary'])
    analysis['audience_insights']['journalists_producers'] = convert_markdown_to_html(analysis['audience_insights']['journalists_producers'])
    analysis['audience_insights']['output_editors'] = convert_markdown_to_html(analysis['audience_insights']['output_editors'])
    analysis['audience_insights']['marketing_teams'] = convert_markdown_to_html(analysis['audience_insights']['marketing_teams'])
    analysis['insight_types']['content_strategy'] = convert_markdown_to_html(analysis['insight_types']['content_strategy'])
    analysis['insight_types']['client_engagement'] = convert_markdown_to_html(analysis['insight_types']['client_engagement'])
    analysis['insight_types']['market_positioning'] = convert_markdown_to_html(analysis['insight_types']['market_positioning'])
    analysis['recommendations'] = convert_markdown_to_html(analysis['recommendations'])
    
    # Render the template with the analysis
    return render_template('ai_analysis.html',
                          session_id=session_id,
                          analysis=analysis,
                          loading=False)

@app.route('/download_ai_analysis/<session_id>/<channel_name>')
def download_ai_analysis(session_id, channel_name):
    """Download the AI-powered analysis as a PDF
    
    This function generates a PDF version of the AI-powered analysis for a specific
    channel and provides it for download.
    
    Args:
        session_id (str): Unique session ID for the current analysis
        channel_name (str): Name of the channel being analyzed
        
    Returns:
        The PDF file for download or a redirect to generate the analysis if not found
    """
    # Check if session exists
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    if not os.path.exists(session_dir):
        flash('Session not found')
        return redirect(url_for('index'))
    
    # Check if analysis exists
    analysis_file = os.path.join(session_dir, f'ai_analysis_{channel_name}.json')
    if not os.path.exists(analysis_file):
        # Start the analysis
        return redirect(url_for('generate_ai_analysis', session_id=session_id, channel_name=channel_name))
    
    # Load the analysis
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Convert markdown to HTML for better formatting in PDF
    analysis_content = {
        'executive_summary': convert_markdown_to_html(analysis['executive_summary']),
        'audience_insights': {
            'journalists_producers': convert_markdown_to_html(analysis['audience_insights']['journalists_producers']),
            'output_editors': convert_markdown_to_html(analysis['audience_insights']['output_editors']),
            'marketing_teams': convert_markdown_to_html(analysis['audience_insights']['marketing_teams'])
        },
        'insight_types': {
            'content_strategy': convert_markdown_to_html(analysis['insight_types']['content_strategy']),
            'client_engagement': convert_markdown_to_html(analysis['insight_types']['client_engagement']),
            'market_positioning': convert_markdown_to_html(analysis['insight_types']['market_positioning'])
        },
        'recommendations': convert_markdown_to_html(analysis['recommendations'])
    }
    
    # Generate a PDF file
    try:
        # Import necessary libraries
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        # Create a PDF file
        pdf_file = os.path.join('static', 'reports', f'{session_id}_{channel_name}_AI_Analysis.pdf')
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        
        # Get styles
        styles = getSampleStyleSheet()
        # Modify existing styles instead of adding new ones
        styles['Heading1'].fontSize = 16
        styles['Heading1'].spaceAfter = 12
        styles['Heading2'].fontSize = 14
        styles['Heading2'].spaceAfter = 8
        styles['Normal'].fontSize = 10
        styles['Normal'].spaceAfter = 6
        
        # Create the content
        content = []
        
        # Title
        content.append(Paragraph(f"AI-Powered Analysis for {channel_name}", styles['Heading1']))
        content.append(Paragraph(f"Generated on {analysis['timestamp']}", styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        # Executive Summary
        content.append(Paragraph("Executive Summary", styles['Heading2']))
        content.append(Paragraph(analysis_content['executive_summary'], styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        # Insights by Audience
        content.append(Paragraph("Insights by Audience", styles['Heading2']))
        
        content.append(Paragraph("For Journalists and Producers", styles['Heading2']))
        content.append(Paragraph(analysis_content['audience_insights']['journalists_producers'], styles['Normal']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph("For Output Editors", styles['Heading2']))
        content.append(Paragraph(analysis_content['audience_insights']['output_editors'], styles['Normal']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph("For Marketing and Client-Facing Teams", styles['Heading2']))
        content.append(Paragraph(analysis_content['audience_insights']['marketing_teams'], styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        # Insights by Type
        content.append(Paragraph("Insights by Type", styles['Heading2']))
        
        content.append(Paragraph("Content Strategy Insights", styles['Heading2']))
        content.append(Paragraph(analysis_content['insight_types']['content_strategy'], styles['Normal']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph("Client Engagement Insights", styles['Heading2']))
        content.append(Paragraph(analysis_content['insight_types']['client_engagement'], styles['Normal']))
        content.append(Spacer(1, 0.1*inch))
        
        content.append(Paragraph("Market Positioning Insights", styles['Heading2']))
        content.append(Paragraph(analysis_content['insight_types']['market_positioning'], styles['Normal']))
        content.append(Spacer(1, 0.25*inch))
        
        # Recommendations
        content.append(Paragraph("Actionable Recommendations", styles['Heading2']))
        content.append(Paragraph(analysis_content['recommendations'], styles['Normal']))
        
        # Build the PDF
        doc.build(content)
        
        # Return the PDF file
        return send_from_directory('static/reports', f'{session_id}_{channel_name}_AI_Analysis.pdf', as_attachment=True)
        
    except Exception as e:
        flash(f"Error generating PDF: {str(e)}")
        return redirect(url_for('view_ai_analysis', session_id=session_id, channel_name=channel_name))

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='RVN Client Analysis Tool')
    parser.add_argument('--port', type=int, default=5019, help='Port to run the server on')
    args = parser.parse_args()
    
    # Create HTML templates if they don't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Reuters Teletrax Analysis</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/tr_styles.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reuters Teletrax Analysis Tool</h1>
            <p class="lead">Upload your Teletrax data export to generate insights and visualizations</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-warning">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h4>Upload Teletrax Data</h4>
                    </div>
                    <div class="card-body">
                        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="file" class="form-label">Select Excel or CSV file</label>
                                <input type="file" class="form-control" id="file" name="file" accept=".xlsx,.xls,.csv">
                                <div class="form-text">Upload your Teletrax data export file (Excel or CSV format)</div>
                            </div>
                            <button type="submit" class="btn btn-primary">Upload and Analyze</button>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h4>Expected Data Format</h4>
                    </div>
                    <div class="card-body">
                        <p>The file should contain the following columns:</p>
                        <ul>
                            <li>Channel: Name</li>
                            <li>Market: Name</li>
                            <li>UTC detection start</li>
                            <li>Local detection start</li>
                            <li>Story ID</li>
                            <li>Slug line</li>
                            <li>Actual detection length</li>
                        </ul>
                        <p>Additional columns may be present but are not required.</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; Reuters 2025 - Teletrax Analysis Tool</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            ''')
    
    if not os.path.exists('templates/analyze.html'):
        with open('templates/analyze.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Reuters Teletrax Analysis</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/tr_styles.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reuters Teletrax Analysis Tool</h1>
            <p class="lead">Analysis Dashboard</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-warning">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h4>Data Overview</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h5>Dataset Statistics</h5>
                                <ul>
                                    <li><strong>Total Records:</strong> {{ stats.total_records }}</li>
                                    <li><strong>Date Range:</strong> {{ stats.date_range }}</li>
                                    <li><strong>Channels:</strong> {{ stats.channels }}</li>
                                    <li><strong>Markets:</strong> {{ stats.markets }}</li>
                                    <li><strong>Unique Stories:</strong> {{ stats.unique_stories }}</li>
                                    <li><strong>Unique Story IDs:</strong> {{ stats.unique_story_ids }}</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h5>Top 5 Stories</h5>
                                <ul>
                                    {% for story, count in top_stories.items() %}
                                        <li><strong>{{ story }}:</strong> {{ count }} detections</li>
                                    {% endfor %}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h4>Available Analyses</h4>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Time Series Analysis</h5>
                                        <p class="card-text">Analyze detection trends over time</p>
                                        <a href="{{ url_for('generate_analysis', analysis_type='time_series', session_id=session_id) }}" class="btn btn-primary">Generate</a>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Top Stories Analysis</h5>
                                        <p class="card-text">Identify the most frequently detected stories</p>
                                        <a href="{{ url_for('generate_analysis', analysis_type='top_stories', session_id=session_id) }}" class="btn btn-primary">Generate</a>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Detection Patterns</h5>
                                        <p class="card-text">Analyze patterns by hour of day and day of week</p>
                                        <a href="{{ url_for('generate_analysis', analysis_type='detection_patterns', session_id=session_id) }}" class="btn btn-primary">Generate</a>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Detection Lengths</h5>
                                        <p class="card-text">Analyze the distribution of detection lengths</p>
                                        <a href="{{ url_for('generate_analysis', analysis_type='detection_lengths', session_id=session_id) }}" class="btn btn-primary">Generate</a>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Channel Comparison</h5>
                                        <p class="card-text">Compare usage patterns between channels</p>
                                        <a href="{{ url_for('generate_analysis', analysis_type='channel_comparison', session_id=session_id) }}" class="btn btn-primary">Generate</a>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4 mb-3">
                                <div class="card">
                                    <div class="card-body">
                                        <h5 class="card-title">Generate All</h5>
                                        <p class="card-text">Generate all analyses and create PowerPoint</p>
                                        <a href="{{ url_for('generate_all_analyses', session_id=session_id) }}" class="btn btn-success">Generate All</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h4>Download Reports</h4>
                    </div>
                    <div class="card-body">
                        <p>After generating all analyses, you can download the PowerPoint presentation:</p>
                        <a href="{{ url_for('download_report', session_id=session_id) }}" class="btn btn-primary">Download PowerPoint</a>
                        <a href="{{ url_for('index') }}" class="btn btn-secondary ms-2">Upload New Data</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; Reuters 2025 - Teletrax Analysis Tool</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            ''')
    
    if not os.path.exists('templates/view_analysis.html'):
        with open('templates/view_analysis.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Reuters Teletrax Analysis</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/tr_styles.css') }}">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Reuters Teletrax Analysis</h1>
            <p class="lead">{{ analysis_title }}</p>
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="alert alert-warning">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h4>{{ analysis_title }}</h4>
                    </div>
                    <div class="card-body text-center">
                        <img src="{{ url_for('static', filename=image_path) }}" class="analysis-image" alt="{{ analysis_title }}">
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h4>Actions</h4>
                    </div>
                    <div class="card-body">
                        <a href="{{ url_for('analyze', session_id=session_id) }}" class="btn btn-primary">Back to Analysis Dashboard</a>
                        <a href="{{ url_for('index') }}" class="btn btn-secondary ms-2">Upload New Data</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; Reuters 2025 - Teletrax Analysis Tool</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            ''')
    
    # Run the app with the specified port
    app.run(debug=True, host='0.0.0.0', port=args.port)
