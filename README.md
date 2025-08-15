# RVN Client Analysis Tool

A suite of tools for analyzing RVN client data to gain insights into client usage patterns, content performance, and regional trends. The application features Thomson Reuters branding throughout the interface and visualizations.

## Overview

This project provides tools to analyze client data, which tracks the usage of Reuters video content across different channels, countries, and networks. The tools help journalists and producers quickly identify patterns and trends in the data without having to manually analyze large spreadsheets.

## Features

- **Data Analysis**: Analyze client data to generate insights on video performance, client engagement, and regional usage trends
- **AI-Powered Insights**: Generate advanced AI analysis using Claude Sonnet 4 for deeper insights and recommendations
- **Visualizations**: Create charts and graphs to visualize the data
- **PowerPoint Generation**: Automatically generate PowerPoint presentations with key findings
- **Single-Slide PowerPoint**: Create channel-specific single-slide PowerPoints for quick presentations
- **Web Interface**: Upload and analyze client data through a user-friendly web interface with Thomson Reuters branding
- **Story ID + Headline Organization**: View stories organized by both Story ID + Headline and by Slug Line for better identification
- **Interactive Visualizations**: Explore data through interactive HTML visualizations
- **Thomson Reuters Branding**: Consistent application of Thomson Reuters brand colors across the interface, visualizations, and PowerPoint presentations

## Components

The project consists of the following components:

1. **teletrax_analysis.py**: Core analysis script that processes client data and generates visualizations and reports
2. **create_sample_data.py**: Script to generate sample client data for testing
3. **teletrax_web_app.py**: Web application for uploading and analyzing client data
4. **static/css/tr_styles.css**: Thomson Reuters branding styles for the web interface

## Thomson Reuters Branding

The application implements the official Thomson Reuters brand colors and typography throughout:

### Brand Colors

- **Primary Colors**:
  - TR Orange (#D64000): Used for primary buttons, headers, and key UI elements
  - TR Racing Green (#123015): Used for secondary elements and footer
  - TR White (#FFFFFF): Used as the main background color

- **Secondary Colors**:
  - Sky Pair (Light: #E3F1FD, Dark: #0874E3): Used for card backgrounds, content areas, and data visualization
  - Additional secondary color pairs are available for specific visualization needs

### Logo Usage

- **Primary Logo**: The full Thomson Reuters logo is used in the header for primary branding
- **Symbol Mark**: The Thomson Reuters symbol mark (circular orange favicon) is used in the footer for a cleaner, more aesthetically pleasing look while maintaining brand identity

### Typography

The application uses the official Thomson Reuters Clario font family:

- **Clario Font**: The exclusive typeface created for Thomson Reuters, designed for balance, clarity, and modernity
  - Used throughout the web application with appropriate weights and sizes
  - Font weights follow the Reuters style guide:
    - Medium weight for headlines and large titles
    - Regular weight for body copy and captions
  - Proper font hierarchy is implemented according to the style guide:
    - Eyebrow = 1.5X body size
    - XL Headline = 8X body size
    - Headline 1 = 6X body size
    - Headline 2 = 3X body size
    - Subhead = 2X body size
    - Body = 1X (base size)

- **Arial Font**: Used as a fallback and for PowerPoint presentations
  - PowerPoint presentations use Arial as recommended by the style guide for downloadable and shareable content
  - Ensures compatibility when users don't have Clario installed

The branding is consistently applied across:
- Web interface elements (buttons, cards, headers)
- Data visualizations and charts
- PowerPoint presentations and slides
- Interactive HTML visualizations

This ensures a professional, cohesive look that aligns with Thomson Reuters brand guidelines for both color and typography.

## Requirements

- Python 3.6+
- Required Python packages:
  - pandas
  - matplotlib
  - plotly
  - python-pptx
  - flask
  - openpyxl

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/rvn-client-analysis.git
cd rvn-client-analysis
```

2. Install the required packages:
```
pip install pandas matplotlib plotly python-pptx flask openpyxl python-dotenv
```

3. Set up environment variables:
   - Create a `.env` file in the root directory based on the provided `.env.example`
   - Add your LiteLLM API key to the `.env` file:
   ```
   LITELLM_API_KEY=your_api_key_here
   LITELLM_API_URL=https://litellm.int.thomsonreuters.com
   ```

## Usage

### Command Line Analysis

To analyze client data from the command line:

1. Place your client data Excel export in the `Sample_Data` directory
2. Run the analysis script:
```
python teletrax_analysis.py
```
3. The script will generate visualizations and a PowerPoint presentation in the `output` directory

### Web Interface

To use the web interface:

1. Start the web application:
```
python teletrax_web_app.py
```
2. Open a web browser and navigate to `http://localhost:5006`
3. Upload your client data Excel export
4. Use the web interface to generate and view different analyses:
   - Generate individual analyses (Time Series, Top Stories, etc.)
   - Create channel-specific single-slide PowerPoints
   - Generate all analyses at once with the "Generate All" button
5. Download options:
   - Download the complete PowerPoint presentation with all visualizations
   - Download channel-specific single-slide PowerPoints

## Data Format

The tool expects client data in Excel format with the following columns:

- Channel: Name
- Market: Name
- UTC detection start
- Local detection start
- Story ID
- Slug line
- Actual detection length

Additional columns may be present but are not required.

### Reuters Video Terminology

The tool handles specific Reuters video terminology to improve clarity in analysis:

- **ADVISORY**: In Reuters video content, the "ADVISORY" prefix in headlines and slug lines indicates a live broadcast. The tool replaces "ADVISORY" with "LIVE:" in the displayed results to make this clearer for users who may not be familiar with Reuters terminology.

## Analysis Types

The tool provides the following types of analysis:

1. **Time Series Analysis**: Track detection trends over time
2. **Top Stories Analysis**: Identify the most frequently detected stories
   - **By Story ID + Headline**: Organize stories by their unique ID and headline for precise identification
   - **By Slug Line**: Organize stories by their slug line for traditional identification
3. **Detection Patterns**: Analyze patterns by hour of day and day of week
4. **Detection Lengths**: Analyze the distribution of detection lengths
5. **Channel Comparison**: Compare usage patterns between channels
6. **Single-Slide Channel Analysis**: Generate a focused single-slide PowerPoint for a specific channel
7. **AI-Powered Analysis**: Generate detailed insights and recommendations using Claude Sonnet 4
   - **Audience-Specific Insights**: Tailored for journalists, editors, and marketing teams
   - **Insight Categories**: Content strategy, client engagement, and market positioning
   - **Actionable Recommendations**: Specific suggestions based on data patterns
   - **All Channels Analysis**: Holistic analysis across all channels with cross-channel comparisons

## Single-Slide PowerPoint Feature

The single-slide PowerPoint feature creates a concise, presentation-ready slide for a specific channel. Each slide contains:

1. **Time Series Chart**: Shows video usage over time with a trend line
2. **Country Distribution**: Pie chart showing video usage by country
3. **Text Summary**: Key statistics and trend information
4. **Major Stories**: Top topics with percentage breakdowns

This feature is particularly useful for:
- Quick presentations to editorial teams
- Channel-specific reporting
- Executive summaries
- Comparing usage patterns between different channels

### Client-Friendly Slide Generation

The tool offers two distinct presentation options:

- **Regular Version**: Contains factual analysis of the client's usage patterns with comprehensive data
- **Client-Friendly Version**: Focuses on positive trends and growth periods, with AI-generated text that emphasizes partnership value

The client-friendly version uses LiteLLM integration to:
- Identify optimal timeframes that show positive growth trends in the time series analysis
- Generate text that highlights positive aspects of the client's usage of Reuters content
- Present data in a way that emphasizes the value of the Reuters-client partnership
- Maintain factual accuracy while focusing on constructive insights

This feature is particularly valuable for:
- Client-facing meetings and presentations
- Business development and relationship management
- Renewal discussions and partnership reviews
- Marketing and sales presentations

### Using the Single-Slide Feature

In the web interface:
1. Navigate to the "Single-Slide PowerPoint (Channel-Specific)" section
2. Either:
   - Select a channel from the dropdown and click "Generate Single-Slide PowerPoint"
   - Use the quick links to generate a slide for a specific channel
3. Toggle the "Client-Friendly Version" option if you want to generate a slide optimized for client presentations
4. Once generated, download the PowerPoint using the provided download buttons

## AI-Powered Analysis Feature

The AI-powered analysis feature uses Claude Sonnet 4 to generate detailed insights and recommendations based on the Teletrax data. This feature provides a deeper level of analysis than traditional visualizations alone.

### Key Capabilities

- **Raw Data Analysis**: The system now sends the complete raw data to the LLM, not just pre-processed summaries, enabling:
  - More nuanced analysis of patterns and correlations
  - Discovery of insights that might be missed in aggregated data
  - Direct examination of individual records to identify outliers and anomalies
  - Custom aggregations and calculations performed by the AI itself

- **Audience-Specific Insights**: Generates tailored insights for different stakeholders:
  - Journalists and Producers: Content performance and story selection guidance
  - Output Editors: Programming and scheduling recommendations
  - Marketing and Client-Facing Teams: Client engagement strategies and market positioning

- **Insight Categories**:
  - Content Strategy Insights: Recommendations for content creation and curation
  - Client Engagement Insights: Patterns in client usage and opportunities for improvement
  - Market Positioning Insights: Competitive analysis and market trends

- **Actionable Recommendations**: Provides specific, actionable recommendations based on data patterns

- **PDF Export**: Download the AI analysis as a formatted PDF report for sharing with stakeholders

### Using the AI Analysis Feature

In the web interface:
1. Navigate to the "Available Analyses" section
2. Find the "AI-Powered Insights" card
3. Select a channel from the dropdown (or choose "All Channels" for a holistic analysis) and click "Generate"
4. The system will process the data and generate the AI analysis
5. View the Executive Summary for a high-level overview
6. Select one of the three deep-dive analysis options to generate specialized insights
7. Download the analysis as a PDF if needed

#### Redesigned Interface

The AI analysis interface has been redesigned for a more focused, user-friendly experience:
- **Executive Summary**: Loads immediately when the analysis is generated
- **Deep-Dive Analysis Options**: Three prominent buttons for specialized analyses
- **On-Demand Generation**: Each deep-dive analysis is generated only when requested
- **Focused Content**: Each analysis type focuses on a specific aspect of the data

#### Deep-Dive Analysis

The deep-dive analysis feature provides specialized, in-depth analyses for specific aspects of the data:
- **Client Relationship Analysis**: Detailed examination of the client's usage patterns and engagement with Reuters content
- **Industry Context Analysis**: Analysis of how the client's usage compares to industry trends and standards
- **Quantitative Analysis**: In-depth statistical analysis of the client's usage data with detailed metrics
- **Editorial Insights**: Specialized analysis for producers and editors with programming recommendations and content performance analysis
- **Marketing Insights**: Client relationship analysis for marketing and business development teams with value proposition and contract renewal support

These deep-dive analyses are generated on-demand using the LiteLLM API, providing additional context and insights beyond the initial analysis. This feature is particularly useful for:
- Preparing for client meetings with comprehensive background information
- Developing targeted strategies for specific clients or content areas
- Understanding complex patterns that require more detailed explanation
- Generating specialized reports for different stakeholders
- Supporting editorial decision-making with data-driven insights
- Enhancing marketing and business development efforts with client-specific analysis

#### All Channels Analysis

The "All Channels" option provides a comprehensive analysis across all channels in your dataset:
- Identifies patterns and trends that span multiple channels
- Compares content preferences between different channels
- Highlights opportunities for cross-channel content strategies
- Provides insights that might not be apparent when looking at individual channels
- Offers recommendations that consider the entire content ecosystem

## Output

The tool generates the following outputs:

- Static images (PNG format)
- Interactive visualizations (HTML format)
- Comprehensive PowerPoint presentation with all analyses
- Channel-specific single-slide PowerPoint presentations

## Documentation

Comprehensive documentation for the RVN Client Analysis Tool is available in the `docs` directory:

- **User Documentation**: Instructions for using the tool, including uploading data, generating analyses, and downloading reports
- **Typography Guide**: Detailed information about the typography implementation, including font files, weights, and hierarchy
- **Branding Guide**: Detailed information about the branding implementation, including color palette, UI elements, and accessibility considerations

### Building the Documentation

The documentation is built using MkDocs with the Material theme. To build and view the documentation locally:

1. Install the required dependencies:
```
pip install -r docs/requirements.txt
```

2. Build and serve the documentation:
```
mkdocs serve
```

3. Open a web browser and navigate to `http://localhost:8000`

### Documentation Structure

- `docs/index.md` - Documentation home page
- `docs/typography.md` - Typography guide
- `docs/branding.md` - Branding guide
- `docs/stylesheets/` - Custom CSS for the documentation site
- `docs/assets/` - Assets for the documentation site

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Reuters for providing the Teletrax data
- The open-source community for the libraries used in this project
