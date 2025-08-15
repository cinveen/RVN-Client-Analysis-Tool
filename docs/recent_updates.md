# Recent Updates

This document provides a chronological list of recent updates, enhancements, and bug fixes to the RVN Client Analysis Tool. It serves as a changelog to help users and developers stay informed about the latest changes to the application.

## August 2025

### August 15, 2025

#### AI Analysis Deep-Dive Feature

- **New Feature**: Added "Learn More" buttons to the AI analysis page for on-demand deep-dive analyses
- **Enhancement**: Users can now request specialized deep-dive analyses for specific aspects of the data
- **Technical Details**:
  - Added "Learn More" buttons to each section of the AI analysis page
  - Implemented AJAX requests to generate deep-dive analyses on demand
  - Created a new `/deep_dive/<category>/<session_id>/<channel_name>` endpoint in `teletrax_web_app.py`
  - Enhanced the LiteLLM client to support specialized deep-dive prompts
- **Deep-Dive Categories**:
  - Client Relationship Analysis: Detailed examination of client usage patterns and engagement
  - Industry Context Analysis: Analysis of how client usage compares to industry trends
  - Quantitative Analysis: In-depth statistical analysis of usage data
  - Temporal Trends Analysis: Detailed examination of usage changes over time
  - Recommendation Details: Expanded recommendations with implementation strategies
- **Impact**: Users can now access more detailed, specialized analyses without regenerating the entire analysis
- **User Experience**: Provides a more interactive, exploratory experience with the AI analysis feature

#### Raw Data Analysis Enhancement

- **Enhancement**: Enhanced the AI analysis to process and analyze the complete raw data, not just pre-processed summaries
- **Technical Details**:
  - Modified `teletrax_web_app.py` to send the complete raw data records to the LLM
  - Updated the system prompt in `litellm_client.py` to instruct the AI to analyze the raw data directly
  - Added explicit instructions for the AI to find patterns, trends, and correlations in the raw data
  - Enabled the AI to perform its own aggregations and calculations rather than relying solely on pre-processed summaries
- **Impact**: AI-generated insights are now more nuanced and can discover patterns that might be missed in aggregated data
- **User Experience**: Provides deeper, more detailed analysis with insights derived directly from individual records


#### AI Analysis Contextual Awareness Enhancement

- **Enhancement**: Enhanced the AI analysis with strict data limitations and contextual awareness
- **Bug Fix**: Fixed an issue where the AI was making unsubstantiated claims about Reuters' market position and ignoring geopolitical context
- **Technical Details**:
  - Added a "STRICT DATA LIMITATIONS" section to the system prompt that explicitly prohibits making claims about Reuters' competitive position
  - Added an "INCORPORATE CONTEXTUAL AWARENESS" section that instructs the AI to consider geopolitical, historical, and cultural context
  - Updated the "BALANCED ANALYSIS APPROACH" to ensure all observations are directly grounded in the data
  - Added explicit instructions to avoid making flattering but unsubstantiated claims about Reuters
- **Impact**: AI-generated insights now correctly incorporate relevant contextual factors and avoid making claims that cannot be validated with the data
- **User Experience**: Eliminates misleading statements like "validates Reuters positioning as the go-to source" when the data contains no competitive information

#### AI Analysis Framework Shift

- **Enhancement**: Fundamentally shifted the AI analysis framework to focus on client usage patterns rather than Reuters content strategy
- **Bug Fix**: Fixed an issue where the AI was incorrectly problematizing client preferences as "strategic risks" or "content waste"
- **Technical Details**:
  - Completely restructured the system prompt in `litellm_client.py` to emphasize client-focused insights
  - Added a "CORRECT ANALYTICAL FRAMEWORK" section that explicitly guides the AI to analyze client preferences, not evaluate Reuters' content strategy
  - Added an "AVOID THESE COMMON MISINTERPRETATIONS" section to prevent the AI from framing client preferences as problems
  - Updated the "BALANCED ANALYSIS APPROACH" to focus on opportunities to enhance client relationships rather than "fixing problems" with Reuters' content
- **Impact**: AI-generated insights now correctly focus on understanding client preferences and usage patterns to strengthen relationships
- **User Experience**: Eliminates misleading statements like "your content strategy is dangerously narrow" or "89% of your stories are essentially invisible" when these simply reflect client preferences

### August 14, 2025

#### AI Analysis ADVISORY Prefix Handling Fix

- **Bug Fix**: Fixed an issue where the AI analysis was treating "ADVISORY" as a content category rather than recognizing it as a prefix for live broadcasts
- **Enhancement**: Updated the system prompt in the LiteLLM client to explicitly explain that "ADVISORY" indicates a live broadcast
- **Technical Details**:
  - Added clear explanation in the system prompt that "ADVISORY" is synonymous with "LIVE:" and not a content category
  - Enhanced data normalization in `teletrax_web_app.py` to consistently replace "ADVISORY" with "LIVE:" in data sent to the AI model
  - Ensured consistent handling of the prefix across all parts of the analysis pipeline
- **Impact**: AI-generated insights and recommendations now correctly interpret "ADVISORY" content as live broadcasts rather than treating it as a separate content category
- **User Experience**: Eliminates confusing recommendations about "expanding advisory content production" when the AI was actually referring to live broadcasts

#### All Channels AI Analysis Feature

- **New Feature**: Added "All Channels" option to the AI analysis dropdown
- **Enhancement**: Enables holistic analysis across all channels in a single report
- **Technical Details**:
  - Modified the AI analysis generation to handle multi-channel data
  - Enhanced the LiteLLM prompt to focus on cross-channel comparisons and trends
  - Added channel comparison data to the AI input, including channel counts and channel-specific story preferences
  - Updated the UI to include "All Channels" as the first option in the dropdown
- **Impact**: Users can now generate comprehensive insights that span across all channels, identifying patterns that might not be apparent when analyzing channels individually

#### Environment Variable System for API Keys

- **Security Enhancement**: Implemented a more secure approach for handling API keys
- **Technical Details**:
  - Created a `load_env.py` module that loads environment variables from a `.env` file
  - Updated both `teletrax_web_app.py` and `teletrax_analysis.py` to import this module
  - Added `.env.example` file to show the required format without exposing actual credentials
- **Impact**: API keys are now stored in a separate `.env` file which is excluded from version control
- **Security Benefits**:
  - Keeps sensitive credentials out of source code
  - Prevents accidental exposure through version control
  - Makes it easier to use different credentials in different environments

### August 7, 2025

#### Enhanced Channel Comparison Analysis

- **New Feature**: Added multiple view options for Channel Comparison Analysis
- **Enhancement**: Users can now switch between three different ways to view channel comparisons:
  - By Slug Line (original implementation)
  - By Story ID (matches the Top Stories Analysis view)
  - By Master Slug (matches the Top Themes Analysis view)
- **Impact**: Users can now analyze channel preferences with greater flexibility and consistency

### August 6, 2025

#### Multiple File Upload Support

- **New Feature**: Added support for uploading multiple Excel/CSV files at once
- **Enhancement**: The system now combines data from multiple files into a single dataset for analysis
- **Impact**: Users can now analyze data from multiple sources without having to manually merge files

#### AI Analysis Template Enhancement

- **Enhancement**: Improved the AI analysis template to better handle truncated recommendations
- **Bug Fix**: Fixed an issue where truncated recommendations would display as "(0-3 months)\n\n**" instead of a helpful message
- **Impact**: Users will now see a helpful message prompting them to regenerate the analysis when recommendations are truncated

### August 5, 2025

#### AI Analysis Recommendations Fix (Enhanced)

- **Bug Fix**: Fixed an issue where the recommendations section in AI analysis was incomplete or truncated
- **Enhancement**: Significantly improved the extraction of recommendations from the AI model response
- **Technical Details**: 
  - Increased the token limit for AI responses from 4000 to 8000 to ensure complete analysis
  - Enhanced regex patterns to better capture recommendations sections in various formats
  - Added intelligent fallback mechanisms that extract recommendations from numbered lists and bullet points
- **Impact**: Users will now see complete recommendations in all AI analyses, improving the actionability of insights

#### ADVISORY Prefix Handling Enhancement

- **Enhancement**: Updated the handling of "ADVISORY" prefixes in headlines and slug lines
- **Change**: Now replacing "ADVISORY" with "LIVE:" instead of removing it completely
- **Rationale**: "ADVISORY" in Reuters video content indicates a live broadcast, so this change makes it clearer for users who may not be familiar with Reuters terminology
- **Impact**: Improves clarity in analysis outputs and AI-generated insights by preserving the live broadcast context

### August 4, 2025

#### AI-Powered Analysis Integration

- **New Feature**: Integrated Claude Sonnet 4 AI model for advanced Teletrax data analysis
- **Insights**: AI generates detailed insights categorized by audience (journalists/producers, output editors, marketing teams)
- **Analysis Types**: Provides content strategy insights, client engagement insights, and market positioning insights
- **Recommendations**: Delivers actionable recommendations based on data patterns and trends
- **PDF Export**: Added ability to download AI analysis as a formatted PDF report
- **Channel-Specific**: Generates tailored analysis for specific channels to identify unique patterns and opportunities

### August 2, 2025

#### PowerPoint Layout Improvements

- **Bug Fix**: Resolved layout issues in PowerPoint presentations where text was overlapping with charts or running off slides
- **Enhancement**: Implemented adaptive layout system that dynamically positions and sizes elements based on content
- **Consistency**: Applied consistent typography and formatting across all slides, particularly in the Key Insights slide

#### Top Themes Analysis

- **Feature Update**: Replaced "Top Stories by Slug Line" with "Top Themes" analysis
- **Enhancement**: Modified analysis to extract and count master slugs (portion before the forward slash)
- **UI Update**: Updated descriptions and labels to explain the concept of master slugs
- **Visualization**: Added new visualization showing the most frequently covered thematic areas

### August 1, 2025

#### Name Change: Reuters Teletrax Analysis to RVN Client Analysis

- **Branding Update**: Changed application name from "Reuters Teletrax Analysis" to "RVN Client Analysis" throughout the application
- **File Updates**: Updated PowerPoint output filenames to reflect the new name
- **Documentation**: Updated all documentation to reflect the new name

## July 2025

### July 15, 2025

#### Thomson Reuters Branding Update

- **UI Update**: Updated all interface elements to align with the latest Thomson Reuters brand guidelines
- **Typography**: Implemented the complete Clario font family with proper hierarchy
- **Color Palette**: Refined color usage throughout the application for better consistency and accessibility

### July 5, 2025

#### Initial Release

- **Core Feature**: Basic Teletrax data analysis capabilities
- **Visualization**: Time series, top stories, and channel comparison visualizations
- **Export**: PowerPoint generation with Thomson Reuters branding
- **Web Interface**: Simple upload and analysis interface
