import os
import json
import requests
import time
import re
from typing import Dict, Any, List, Optional

class LiteLLMClient:
    """
    Client for interacting with the LiteLLM API to generate AI-powered analysis
    of Teletrax data using Claude Sonnet 4.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialize the LiteLLM client with API credentials.
        
        Args:
            api_key: LiteLLM API key. If not provided, will look for LITELLM_API_KEY env var
            api_url: LiteLLM API URL. If not provided, will look for LITELLM_API_URL env var
        """
        self.api_key = api_key or os.environ.get('LITELLM_API_KEY')
        self.api_url = api_url or os.environ.get('LITELLM_API_URL', 'https://litellm.int.thomsonreuters.com')
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it as an argument or set LITELLM_API_KEY environment variable.")
    
    def generate_analysis(self, 
                         teletrax_data: Dict[str, Any], 
                         channel_name: str,
                         max_retries: int = 3,
                         retry_delay: int = 2) -> Dict[str, Any]:
        """
        Generate AI-powered analysis of Teletrax data using Claude Sonnet 4.
        
        Args:
            teletrax_data: Dictionary containing processed Teletrax data
            channel_name: Name of the channel being analyzed
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            Dictionary containing the generated analysis
        """
        # Prepare the prompt with the Teletrax data
        prompt = self._prepare_prompt(teletrax_data, channel_name)
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ],
            "temperature": 0.3,  # Lower temperature for more factual responses
            "max_tokens": 15000  # Increased to prevent truncation of analysis
        }
        
        # Make the API request with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120  # Longer timeout for detailed analysis
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract and structure the analysis
                analysis = self._structure_analysis(result, channel_name)
                return analysis
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to generate analysis after {max_retries} attempts: {str(e)}")
    
    def _prepare_prompt(self, teletrax_data: Dict[str, Any], channel_name: str) -> Dict[str, str]:
        """
        Prepare the prompt for the LLM with Teletrax data.
        
        Args:
            teletrax_data: Dictionary containing processed Teletrax data and raw data
            channel_name: Name of the channel being analyzed
            
        Returns:
            Dictionary containing system and user prompts
        """
        # System prompt with instructions for the LLM
        system_prompt = """
You are a world-class data analytics expert, hired by Reuters to provide objective, client-focused insights based on story usage data. Your primary goal is to understand client preferences and usage patterns to help Reuters strengthen client relationships. When a user provides you with Teletrax data, be as detailed as possible in your analysis. Provide a comprehensive report that helps Reuters better understand and serve these specific clients. Max out your tokens. Find the most nuanced details you can, see the signal in the noise.

ANALYZE THE RAW DATA DIRECTLY:
You will be provided with both raw data records and pre-processed summaries. Your task is to:
1. Analyze the raw data directly to find patterns, trends, and insights that might not be captured in the pre-processed summaries
2. Look for correlations between different variables (e.g., detection time, story type, detection length)
3. Identify any outliers or anomalies in the data that might indicate unique client preferences
4. Perform your own aggregations and calculations rather than relying solely on the provided summaries
5. Use the raw data to validate or challenge the patterns shown in the pre-processed summaries

The raw data includes these key fields:
- Channel: Name - The name of the channel that used the content
- Market: Name - The market/country where the channel is based
- UTC detection start - When the content was detected (in UTC time)
- Local detection start - When the content was detected (in local time)
- Story ID - Unique identifier for the story
- Slug line - Description of the story content
- Headline - The headline of the story
- Detection Length (seconds) - How long the content was used
- Topic/Subtopic - Categorization of the content
- Detection Year/Month/Day/Hour/Weekday - Temporal breakdown of detection time

CRITICAL DATA CONTEXT - READ CAREFULLY:
The data you are analyzing represents ONLY the detections from the specific channels included in the upload. It does NOT represent all Reuters content production or global usage patterns. This means:

1. This data shows ONLY what these specific clients chose to use from Reuters' offerings
2. The data does NOT show what Reuters produced or offered - only what these clients selected
3. High percentages for certain content types (e.g., "58% Israel-Palestine content") reflect CLIENT PREFERENCES, not Reuters' content strategy
4. Stories with fewer detections are still being used by clients - they just have different usage patterns
5. The data cannot tell you anything about Reuters' overall content production, strategy, or quality
6. When you see "73% of detections" or similar statistics, this means "73% of what these specific clients chose to use"

STRICT DATA LIMITATIONS - NEVER MAKE THESE CLAIMS:
1. NEVER claim Reuters is "the go-to source" or "leader" for any type of content - you have NO competitive data
2. NEVER make claims about Reuters' market position relative to competitors - the data shows nothing about competitors
3. NEVER state that certain content "validates Reuters' positioning" - you cannot validate positioning without competitive data
4. NEVER claim to know what percentage of a channel's total news content comes from Reuters - you only see Reuters content
5. NEVER make claims about the "value" or "quality" of Reuters content - you can only observe usage patterns
6. NEVER suggest you know why a client isn't using certain Reuters content - you only see what they did use, not what they rejected
7. NEVER claim to know a client's overall content strategy - you only see their Reuters usage, which may be a small part of their total content

INCORPORATE CONTEXTUAL AWARENESS:
When analyzing client preferences, consider relevant geopolitical, historical, and cultural context:
1. For news channels based in specific countries, consider their national interests (e.g., a US network's interest in Middle East conflicts relates to US foreign policy)
2. For channels with specific audience demographics, consider their audience's likely interests
3. Consider the time period of the data and what major world events were occurring then
4. Recognize that high usage of conflict coverage by a channel in a region directly affected by that conflict is expected and logical
5. Understand that channels from countries with historical ties to certain regions will naturally have higher interest in those regions
6. Consider a channel's known political orientation or focus (business, general news, etc.) when analyzing their content preferences

CORRECT ANALYTICAL FRAMEWORK:
Your analysis should focus on understanding client preferences and usage patterns, NOT evaluating Reuters' content strategy. The correct framework is:

1. What types of content do these specific clients prefer? (Not "Reuters' content strategy is too narrow")
2. When and how do these clients use Reuters content? (Not "Reuters is missing opportunities")
3. What patterns exist in these clients' usage that might inform better service? (Not "Reuters has poor story selection")
4. How might Reuters better serve these specific clients based on their demonstrated preferences? (Not "Reuters needs to fix its content strategy")

AVOID THESE COMMON MISINTERPRETATIONS:
1. DO NOT frame high percentages of certain content types as "dangerously narrow" or "strategic risks" - they simply reflect what these clients chose to use
2. DO NOT describe stories with fewer detections as "invisible," "poor story selection," or "content waste" - they're still being used, just at different rates
3. DO NOT characterize client concentration as "dependency risks" - the data only shows which channels were included in the analysis
4. DO NOT make assumptions about Reuters' overall content strategy based on what clients chose to use
5. DO NOT assume that patterns in the data represent "problems" that need to be "fixed" - they may simply reflect client preferences
6. DO NOT make flattering but unsubstantiated claims about Reuters just to have something positive to say

BALANCED ANALYSIS APPROACH:
1. Identify clear patterns in how these specific clients use Reuters content
2. Note both consistent usage patterns and any changes or anomalies over time
3. Highlight opportunities to better serve these specific clients based on their demonstrated preferences
4. Provide specific, actionable recommendations to strengthen relationships with these clients
5. Be direct and honest about the data, but avoid catastrophizing or problematizing client preferences
6. Focus on opportunities to enhance client relationships rather than "fixing problems" with Reuters' content
7. Ground ALL observations and recommendations directly in the data - if you can't point to specific numbers that support a claim, don't make it
8. When discussing geopolitical content preferences, acknowledge the relevant historical and political context that likely explains these preferences

EVALUATIVE FRAMEWORK AND INDUSTRY CONTEXT:
Go beyond basic data description to provide evaluative insights:
1. Compare client usage patterns to known industry trends and news consumption patterns
2. Evaluate how effectively current content is meeting client needs based on their usage patterns
3. Assess how geopolitical events during this period influenced content preferences
4. Identify untapped potential in current usage patterns that could be leveraged
5. Consider how client usage compares to similar channels in their market or region

QUANTITATIVE ANALYSIS REQUIREMENTS:
Provide data-driven insights with specific metrics:
1. Calculate and highlight significant percentage changes or trends over time
2. Identify correlations between different variables (e.g., story type and detection length)
3. Use statistical measures to identify outliers and anomalies worth investigating
4. Segment data in meaningful ways to reveal patterns (by time period, content type, etc.)
5. Quantify the strength of observed patterns and trends

CLIENT RELATIONSHIP FOCUS:
For each insight, explain its implications for client relationship management:
1. Identify opportunities to strengthen client relationships based on demonstrated preferences
2. Suggest tailored engagement strategies for different client segments
3. Highlight content types that could deepen client relationships
4. Identify potential pain points or areas of misalignment
5. Recommend metrics to track relationship health over time

TEMPORAL ANALYSIS REQUIREMENTS:
Analyze how patterns change over time:
1. Analyze how client preferences have evolved over the time period covered by the data
2. Identify cyclical patterns (daily, weekly, monthly) and their implications
3. Predict potential future trends based on historical patterns
4. Compare usage during different time periods (e.g., before/after major events)
5. Identify seasonal or event-driven changes in content preferences

SPECIFIC RECOMMENDATIONS FORMAT:
Provide 3-5 highly specific, actionable recommendations with:
1. Clear implementation timeline (immediate, 1-3 months, 3-6 months)
2. Expected outcomes and metrics for measuring success
3. Priority level based on potential impact and implementation difficulty
4. Specific teams or roles that should be responsible for implementation
5. Concrete next steps to begin implementation

Understanding Teletrax and Its Use for Reuters:

Teletrax is a content tracking system that helps Reuters monitor where and how its video content is being used by broadcast and digital clients. It provides real-time analytics on video distribution, usage patterns, and market reach, allowing Reuters to assess the impact and performance of its video content across multiple platforms.

How Teletrax Works:
1. Invisible Watermarking:
   • Teletrax embeds imperceptible digital watermarks into Reuters video content before distribution.
   • These watermarks remain in the video, even if the content is edited, clipped, or re-encoded.
2. Global Monitoring Network:
   • The system continuously scans TV broadcasts, online streams, and digital platforms for Teletrax-embedded watermarks.
   • It detects and reports video usage in near real-time.
3. Data Collection & Analysis:
   • Reports who used the video, when, where, and how often.
   • Captures key metadata, such as airtime, program context, and geographic distribution.
   • Provides insights into client engagement, content performance, and potential underutilization.

Important Reuters Content Terminology:
• "ADVISORY" prefix: When you see "ADVISORY" at the beginning of a slug line or headline, this indicates a LIVE broadcast. It is NOT a content category or type. In the data processing, "ADVISORY" is replaced with "LIVE:" for clarity. For example, "ADVISORY USA-ELECTION/" and "LIVE: USA-ELECTION/" refer to the same content - a live broadcast of election coverage. Do not treat "ADVISORY" content as a separate category in your analysis or recommendations.

Why Reuters Uses Teletrax:
• Client Engagement: Understand how broadcast and digital partners are using Reuters content.
• Performance Metrics: Identify the most and least-used stories to optimize future coverage.
• Commercial Strategy: Align sales efforts with data on client usage patterns.
• Competitive Benchmarking: Compare Reuters' pickup with competitors.
• Editorial Impact: Evaluate whether key stories are reaching their intended audiences.

Your analysis should be structured in the following way:
1. Executive Summary - A brief overview of the key findings
2. Key Insights - 4-6 major insights organized by importance, each including:
   - The data-backed finding
   - Industry context
   - Client relationship implications
   - Quantitative support
3. Opportunity Analysis:
   - Underutilized content types
   - Emerging client interests
   - Competitive positioning opportunities
   - Content timing optimization
4. Strategic Recommendations - 3-5 specific, actionable recommendations with:
   - Clear implementation timeline
   - Expected outcomes
   - Success metrics
   - Priority level
"""

        # Create a summary of the Teletrax data for the user prompt
        data_summary = self._create_data_summary(teletrax_data, channel_name)
        
        # User prompt with the data summary
        if channel_name == "All Channels":
            user_prompt = f"""
Please analyze the following Teletrax data across all channels and provide detailed insights and recommendations. This analysis should look at the data holistically, comparing and contrasting usage patterns between different channels while also identifying overall trends.

{data_summary}

Based on this data, please provide:
1. An executive summary of key findings across all channels
2. Insights categorized by audience (journalists/producers, output editors, marketing/client-facing teams)
3. Insights categorized by type (content strategy, client engagement, market positioning)
4. Specific, actionable recommendations for Reuters teams

Please be as detailed and specific as possible in your analysis, focusing on:
- Patterns and trends across all channels
- Differences in content preferences between channels
- Opportunities for cross-channel content strategies
- Regional and thematic insights that emerge when looking at all channels together

This should be a holistic analysis that considers all channels together, not separate analyses for each channel.
"""
        else:
            user_prompt = f"""
Please analyze the following Teletrax data for the channel {channel_name} and provide detailed insights and recommendations.

{data_summary}

Based on this data, please provide:
1. An executive summary of key findings
2. Insights categorized by audience (journalists/producers, output editors, marketing/client-facing teams)
3. Insights categorized by type (content strategy, client engagement, market positioning)
4. Specific, actionable recommendations for Reuters teams

Please be as detailed and specific as possible in your analysis, focusing on patterns, trends, and opportunities that might not be immediately obvious.
"""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
    def _create_data_summary(self, teletrax_data: Dict[str, Any], channel_name: str) -> str:
        """
        Create a summary of the Teletrax data for inclusion in the prompt.
        
        Args:
            teletrax_data: Dictionary containing processed Teletrax data
            channel_name: Name of the channel being analyzed
            
        Returns:
            String containing a summary of the Teletrax data
        """
        # Extract key statistics from the data
        stats = teletrax_data.get('stats', {})
        top_stories = teletrax_data.get('top_stories', {})
        
        # Format the top stories
        top_stories_text = "\n".join([f"- {story}: {count} detections" for story, count in top_stories.items()])
        
        # Create the data summary
        summary = f"""
CHANNEL: {channel_name}

DATASET STATISTICS:
- Total Records: {stats.get('total_records', 'N/A')}
- Date Range: {stats.get('date_range', 'N/A')}
- Markets: {stats.get('markets', 'N/A')}
- Unique Stories: {stats.get('unique_stories', 'N/A')}
- Unique Story IDs: {stats.get('unique_story_ids', 'N/A')}

TOP STORIES:
{top_stories_text}
"""
        
        # Add additional data if available
        if 'top_themes' in teletrax_data:
            themes_text = "\n".join([f"- {theme}: {count}" for theme, count in teletrax_data['top_themes'].items()])
            summary += f"\nTOP THEMES:\n{themes_text}\n"
        
        if 'detection_patterns' in teletrax_data:
            patterns = teletrax_data['detection_patterns']
            summary += f"\nDETECTION PATTERNS:\n"
            summary += f"- Peak Detection Hours: {patterns.get('peak_hours', 'N/A')}\n"
            summary += f"- Peak Detection Days: {patterns.get('peak_days', 'N/A')}\n"
        
        if 'country_distribution' in teletrax_data:
            countries_text = "\n".join([f"- {country}: {percentage}%" for country, percentage in teletrax_data['country_distribution'].items()])
            summary += f"\nGEOGRAPHIC DISTRIBUTION:\n{countries_text}\n"
        
        # Add channel comparison data if available (for "All Channels" analysis)
        if 'channel_comparison' in teletrax_data:
            channel_comp = teletrax_data['channel_comparison']
            
            # Add channel counts
            if 'channel_counts' in channel_comp:
                summary += f"\nCHANNEL DISTRIBUTION:\n"
                channel_counts_text = "\n".join([f"- {ch}: {count} detections" for ch, count in channel_comp['channel_counts'].items()])
                summary += f"{channel_counts_text}\n"
            
            # Add channel-specific story preferences
            if 'channel_story_preferences' in channel_comp:
                summary += f"\nCHANNEL-SPECIFIC STORY PREFERENCES:\n"
                for ch, stories in channel_comp['channel_story_preferences'].items():
                    summary += f"\n{ch} Top Stories:\n"
                    ch_stories_text = "\n".join([f"  - {story}: {count} detections" for story, count in stories.items()])
                    summary += f"{ch_stories_text}\n"
        
        return summary
    
    def _structure_analysis(self, api_response: Dict[str, Any], channel_name: str) -> Dict[str, Any]:
        """
        Structure the raw API response into a more usable format.
        
        Args:
            api_response: Raw API response from LiteLLM
            channel_name: Name of the channel being analyzed
            
        Returns:
            Dictionary containing the structured analysis
        """
        try:
            # Extract the content from the API response
            content = api_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Split the content into sections
            sections = self._split_into_sections(content)
            
            # Extract the recommendations directly from the raw content
            import re
            
            # First, try to get it from the sections dictionary
            recommendations = sections.get('Actionable Recommendations', '')
            
            # If that didn't work, try to extract it directly from the raw content using multiple patterns
            if not recommendations or recommendations == "**":
                # Pattern 1: Look for the section with "Actionable Recommendations" heading (case insensitive)
                actionable_match = re.search(r'(?:#+\s*|\d+\.\s*)(?:ACTIONABLE RECOMMENDATIONS|Actionable Recommendations)(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z]|$)', 
                                           content, re.DOTALL | re.IGNORECASE)
                if actionable_match:
                    recommendations = actionable_match.group(1).strip()
                
                # Pattern 2: Look for the section with "Recommendations" heading
                if not recommendations or recommendations == "**":
                    recommendations_match = re.search(r'(?:#+\s*|\d+\.\s*)(?:RECOMMENDATIONS|Recommendations)(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z]|$)', 
                                                   content, re.DOTALL | re.IGNORECASE)
                    if recommendations_match:
                        recommendations = recommendations_match.group(1).strip()
                
                # Pattern 3: Look for the section after "Market Positioning Insights"
                if not recommendations or recommendations == "**":
                    market_pos_match = re.search(r'Market Positioning Insights(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z])', 
                                               content, re.DOTALL | re.IGNORECASE)
                    if market_pos_match:
                        next_section_start = market_pos_match.end()
                        next_section_match = re.search(r'(?:#+\s*|\d+\.\s*)([A-Z][A-Za-z\s]+)', 
                                                     content[next_section_start:], re.IGNORECASE)
                        if next_section_match:
                            section_title = next_section_match.group(1).strip()
                            if "recommendation" in section_title.lower() or "action" in section_title.lower():
                                section_start = next_section_start + next_section_match.start()
                                recommendations_match = re.search(r'(?:#+\s*|\d+\.\s*)' + re.escape(section_title) + r'(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z]|$)', 
                                                               content[section_start:], re.DOTALL | re.IGNORECASE)
                                if recommendations_match:
                                    recommendations = recommendations_match.group(1).strip()
                
                # Pattern 4: Look for the section with "4." heading (assuming recommendations is the 4th section)
                if not recommendations or recommendations == "**":
                    section4_match = re.search(r'(?:#+\s*|^)4\.\s+([A-Z][A-Za-z\s]+)(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z]|$)', 
                                             content, re.DOTALL | re.IGNORECASE)
                    if section4_match:
                        section_title = section4_match.group(1).strip()
                        if "recommendation" in section_title.lower() or "action" in section_title.lower():
                            recommendations = section4_match.group(2).strip()
                
                # Pattern 5: Look for any numbered list items that might be recommendations
                if not recommendations or recommendations == "**":
                    # Look for numbered list items (1., 2., etc.) that might be recommendations
                    numbered_items_match = re.findall(r'\d+\.\s+([^\n]+(?:\n(?!\d+\.).*)*)', content, re.DOTALL)
                    if numbered_items_match and len(numbered_items_match) >= 3:
                        # Take the last set of numbered items (likely recommendations)
                        last_third = numbered_items_match[-int(len(numbered_items_match)/3):]
                        recommendations = "\n".join([f"{i+1}. {item.strip()}" for i, item in enumerate(last_third)])
                
                # Pattern 6: Look for bullet points that might be recommendations
                if not recommendations or recommendations == "**":
                    # Look for bullet points (-, *, •) that might be recommendations
                    bullet_items_match = re.findall(r'(?:^|\n)(?:\s*[-*•]\s+)([^\n]+(?:\n(?![-*•]).*)*)', content, re.DOTALL)
                    if bullet_items_match and len(bullet_items_match) >= 3:
                        # Take the last set of bullet points (likely recommendations)
                        last_third = bullet_items_match[-int(len(bullet_items_match)/3):]
                        recommendations = "\n".join([f"- {item.strip()}" for item in last_third])
                
                # Pattern 7: Last resort - extract everything after the last identified section
                if not recommendations or recommendations == "**":
                    last_section_match = re.search(r'(?:#+\s*|\d+\.\s*)(?:Market Positioning Insights|CLIENT ENGAGEMENT INSIGHTS|CONTENT STRATEGY INSIGHTS)(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z])', 
                                                 content, re.DOTALL | re.IGNORECASE)
                    if last_section_match:
                        last_section_end = last_section_match.end()
                        next_section_match = re.search(r'(?:#+\s*|\d+\.\s*)([A-Z][A-Za-z\s]+)(.*?)(?:$)', 
                                                     content[last_section_end:], re.DOTALL | re.IGNORECASE)
                        if next_section_match:
                            section_title = next_section_match.group(1).strip()
                            if "recommendation" in section_title.lower() or "action" in section_title.lower():
                                recommendations = next_section_match.group(2).strip()
            
            # If we still don't have recommendations or it's just "**", use a more aggressive approach
            if not recommendations or recommendations == "**":
                # Look for any section that might contain recommendations
                recommendation_keywords = ["recommendation", "action", "next step", "strategy", "tactic", "suggest", "advise", "propose"]
                for keyword in recommendation_keywords:
                    keyword_match = re.search(r'(?:#+\s*|\d+\.\s*)(?:[A-Z][A-Za-z\s]*' + re.escape(keyword) + r'[A-Za-z\s]*)(.*?)(?:(?:#+\s*|\d+\.\s*)[A-Z]|$)', 
                                            content, re.DOTALL | re.IGNORECASE)
                    if keyword_match:
                        recommendations = keyword_match.group(1).strip()
                        break
            
            # If all else fails, generate default recommendations based on the content
            if not recommendations or recommendations == "**":
                # Extract key phrases that might indicate recommendations
                recommendation_phrases = []
                
                # Look for phrases like "should", "could", "recommend", etc.
                recommendation_indicators = [
                    r'(?:Reuters|RVN)(?:\s\w+){0,5}\s(?:should|could|must|need to|recommend)(?:\s\w+){1,20}',
                    r'(?:recommend|suggest|advise|propose)(?:\s\w+){1,20}',
                    r'(?:increase|improve|enhance|optimize|focus on|prioritize|consider)(?:\s\w+){1,20}'
                ]
                
                for indicator in recommendation_indicators:
                    matches = re.findall(indicator, content, re.IGNORECASE)
                    recommendation_phrases.extend(matches)
                
                if recommendation_phrases:
                    # Format the extracted phrases as recommendations
                    recommendations = "Based on the analysis, here are key recommendations:\n\n"
                    for i, phrase in enumerate(recommendation_phrases[:5]):  # Limit to top 5
                        recommendations += f"{i+1}. {phrase.strip()}.\n\n"
                else:
                    # Last resort - extract the last 20% of the content as recommendations
                    content_length = len(content)
                    last_fifth_start = int(content_length * 0.8)
                    recommendations = "Recommendations extracted from analysis:\n\n" + content[last_fifth_start:].strip()
            
            # Create the structured analysis
            analysis = {
                'channel_name': channel_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'executive_summary': sections.get('Executive Summary', ''),
                'audience_insights': {
                    'journalists_producers': sections.get('For Journalists and Producers', ''),
                    'output_editors': sections.get('For Output Editors', ''),
                    'marketing_teams': sections.get('For Marketing and Client-Facing Teams', '')
                },
                'insight_types': {
                    'content_strategy': sections.get('Content Strategy Insights', ''),
                    'client_engagement': sections.get('Client Engagement Insights', ''),
                    'market_positioning': sections.get('Market Positioning Insights', '')
                },
                'recommendations': recommendations,
                'raw_content': content  # Include the raw content for reference
            }
            
            return analysis
            
        except Exception as e:
            # If there's an error, return a simplified structure with the raw content
            return {
                'channel_name': channel_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'raw_content': api_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            }
    
    def generate_deep_dive(self, 
                          teletrax_data: Dict[str, Any], 
                          channel_name: str,
                          category: str,
                          max_retries: int = 3,
                          retry_delay: int = 2) -> Dict[str, Any]:
        """
        Generate a specialized deep dive analysis focusing on a specific aspect of the Teletrax data.
        
        Args:
            teletrax_data: Dictionary containing processed Teletrax data
            channel_name: Name of the channel being analyzed
            category: Category of deep dive analysis (e.g., 'client_relationship', 'industry_context')
            max_retries: Maximum number of retry attempts for API calls
            retry_delay: Delay between retry attempts in seconds
            
        Returns:
            Dictionary containing the specialized deep dive analysis
        """
        # Create a specialized prompt based on the category
        specialized_prompt = self._create_specialized_prompt(category, channel_name)
        
        # Create a data summary
        data_summary = self._create_data_summary(teletrax_data, channel_name)
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "anthropic/claude-sonnet-4-20250514",
            "messages": [
                {"role": "system", "content": specialized_prompt},
                {"role": "user", "content": f"Please provide a detailed deep dive analysis on the {category.replace('_', ' ')} aspects of the following Teletrax data for {channel_name}:\n\n{data_summary}"}
            ],
            "temperature": 0.3,
            "max_tokens": 15000  # Maximum tokens for depth
        }
        
        # Make the API request with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120  # Longer timeout for detailed analysis
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract the content from the API response
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                return {
                    'category': category,
                    'channel_name': channel_name,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'content': content
                }
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise Exception(f"Failed to generate deep dive analysis after {max_retries} attempts: {str(e)}")
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """
        Split the content into sections based on headings.
        
        Args:
            content: Raw content from the API response
            
        Returns:
            Dictionary mapping section headings to their content
        """
        import re
        # Common section headings to look for
        section_headings = [
            'Executive Summary',
            'For Journalists and Producers',
            'For Output Editors',
            'For Marketing and Client-Facing Teams',
            'Content Strategy Insights',
            'Client Engagement Insights',
            'Market Positioning Insights',
            'Actionable Recommendations'
        ]
        
        # Initialize the sections dictionary
        sections = {}
        
        # Extract the Executive Summary section
        exec_summary_match = re.search(r'(?:#+\s*(?:\d+\.\s*)?Executive Summary)(.*?)(?:---|\n#+\s*(?:\d+\.\s*)?)', content, re.DOTALL | re.IGNORECASE)
        if exec_summary_match:
            sections['Executive Summary'] = exec_summary_match.group(1).strip()
        
        # Extract the For Journalists and Producers section
        journalists_match = re.search(r'### For Journalists and Producers(.*?)(?:###|---)', content, re.DOTALL | re.IGNORECASE)
        if journalists_match:
            sections['For Journalists and Producers'] = journalists_match.group(1).strip()
        
        # Extract the For Output Editors section
        editors_match = re.search(r'### For Output Editors(.*?)(?:###|---)', content, re.DOTALL | re.IGNORECASE)
        if editors_match:
            sections['For Output Editors'] = editors_match.group(1).strip()
        
        # Extract the For Marketing and Client-Facing Teams section
        marketing_match = re.search(r'### For Marketing and Client-Facing Teams(.*?)(?:---|\n#+\s*(?:\d+\.\s*)?)', content, re.DOTALL | re.IGNORECASE)
        if marketing_match:
            sections['For Marketing and Client-Facing Teams'] = marketing_match.group(1).strip()
        
        # Extract the Content Strategy Insights section
        content_strategy_match = re.search(r'### Content Strategy Insights(.*?)(?:###|---)', content, re.DOTALL | re.IGNORECASE)
        if content_strategy_match:
            sections['Content Strategy Insights'] = content_strategy_match.group(1).strip()
        
        # Extract the Client Engagement Insights section
        client_engagement_match = re.search(r'### Client Engagement Insights(.*?)(?:###|---)', content, re.DOTALL | re.IGNORECASE)
        if client_engagement_match:
            sections['Client Engagement Insights'] = client_engagement_match.group(1).strip()
        
        # Extract the Market Positioning Insights section
        market_positioning_match = re.search(r'### Market Positioning Insights(.*?)(?:---|\n#+\s*(?:\d+\.\s*)?)', content, re.DOTALL | re.IGNORECASE)
        if market_positioning_match:
            sections['Market Positioning Insights'] = market_positioning_match.group(1).strip()
        
        # Extract the Actionable Recommendations section
        actionable_match = re.search(r'(?:#+\s*(?:\d+\.\s*)?Actionable Recommendations)(.*?)(?:---|\n#+\s*(?:\d+\.\s*)?|$)', content, re.DOTALL | re.IGNORECASE)
        if actionable_match:
            sections['Actionable Recommendations'] = actionable_match.group(1).strip()
        
        return sections
    
    def _create_specialized_prompt(self, category: str, channel_name: str) -> str:
        """
        Create a specialized prompt for a specific deep dive category.
        
        Args:
            category: Category of deep dive analysis
            channel_name: Name of the channel being analyzed
            
        Returns:
            Specialized prompt for the deep dive analysis
        """
        # Base context that applies to all specialized prompts
        base_context = """
You are a world-class data analytics expert, hired by Reuters to provide objective, client-focused insights based on story usage data. Your primary goal is to understand client preferences and usage patterns to help Reuters strengthen client relationships.

CRITICAL DATA CONTEXT - READ CAREFULLY:
The data you are analyzing represents ONLY the detections from the specific channels included in the upload. It does NOT represent all Reuters content production or global usage patterns. This means:

1. This data shows ONLY what these specific clients chose to use from Reuters' offerings
2. The data does NOT show what Reuters produced or offered - only what these clients selected
3. High percentages for certain content types reflect CLIENT PREFERENCES, not Reuters' content strategy
4. The data cannot tell you anything about Reuters' overall content production, strategy, or quality

STRICT DATA LIMITATIONS - NEVER MAKE THESE CLAIMS:
1. NEVER claim Reuters is "the go-to source" or "leader" for any type of content - you have NO competitive data
2. NEVER make claims about Reuters' market position relative to competitors - the data shows nothing about competitors
3. NEVER state that certain content "validates Reuters' positioning" - you cannot validate positioning without competitive data
4. NEVER claim to know what percentage of a channel's total news content comes from Reuters - you only see Reuters content
5. NEVER make claims about the "value" or "quality" of Reuters content - you can only observe usage patterns
6. NEVER suggest you know why a client isn't using certain Reuters content - you only see what they did use, not what they rejected
7. NEVER claim to know a client's overall content strategy - you only see their Reuters usage, which may be a small part of their total content
"""

        # Specialized prompts for each category
        if category == 'client_relationship':
            return base_context + """
SPECIALIZED DEEP DIVE: CLIENT RELATIONSHIP ANALYSIS

Analyze the client relationship implications of the Teletrax data in extreme depth. Focus exclusively on:

1. Client Engagement Patterns:
   - Analyze patterns in how this client engages with Reuters content
   - Identify preferences in content types, formats, and timing
   - Determine if engagement is consistent or varies by time period, topic, or other factors
   - Compare this client's engagement patterns with typical patterns for similar clients

2. Relationship Strengthening Opportunities:
   - Identify specific content types or topics that could deepen the relationship
   - Suggest tailored content packages or services based on demonstrated preferences
   - Recommend engagement strategies specific to this client's usage patterns
   - Propose ways to increase the client's utilization of Reuters content

3. Potential Pain Points:
   - Identify any patterns that might indicate dissatisfaction or unmet needs
   - Analyze any declining usage trends and their potential causes
   - Identify content areas where the client might be underserved
   - Suggest proactive measures to address potential issues before they affect the relationship

4. Tailored Engagement Strategies:
   - Develop a detailed client engagement plan with specific touchpoints
   - Suggest personalized content recommendations based on usage history
   - Recommend communication frequency and preferred channels
   - Propose client-specific events, briefings, or other relationship-building activities

5. Relationship Health Metrics:
   - Suggest specific KPIs to track the health of this client relationship
   - Establish baselines and targets for these metrics
   - Recommend a monitoring schedule and response thresholds
   - Propose a framework for regular relationship reviews

Provide extremely detailed analysis with specific examples from the data. Your response should be comprehensive and actionable, with concrete recommendations that Reuters teams can implement immediately.
"""
        elif category == 'industry_context':
            return base_context + """
SPECIALIZED DEEP DIVE: INDUSTRY CONTEXT ANALYSIS

Analyze the Teletrax data in the context of broader industry trends and news consumption patterns. Focus exclusively on:

1. Industry Trend Comparison:
   - Compare the client's usage patterns to known industry trends in news consumption
   - Identify where the client aligns with or diverges from industry norms
   - Analyze how industry shifts might be reflected in the client's content preferences
   - Suggest how Reuters can position its content in light of these industry trends

2. Geopolitical Context:
   - Analyze how major geopolitical events during this period influenced content preferences
   - Identify correlations between news cycles and content usage patterns
   - Assess how regional interests might explain certain content preferences
   - Suggest how Reuters can better align content with geopolitical interests relevant to this client

3. Media Landscape Analysis:
   - Consider how this client's position in the media landscape affects their content needs
   - Analyze how competitive pressures might influence content selection
   - Identify potential gaps in the market that Reuters could help the client address
   - Suggest how Reuters can differentiate its content offering in this context

4. Audience Demographics Influence:
   - Analyze how the client's audience demographics might influence their content selection
   - Identify content preferences that align with their audience's likely interests
   - Suggest content types that might resonate with their specific audience
   - Recommend ways Reuters can help the client better serve their audience

5. Future Trend Prediction:
   - Based on industry analysis, predict future content needs for this client
   - Identify emerging topics or formats that might become important
   - Suggest how Reuters can prepare to meet these future needs
   - Recommend proactive steps to position Reuters as a forward-thinking partner

Provide extremely detailed analysis with specific examples from the data and references to relevant industry trends. Your response should be comprehensive and actionable, with concrete recommendations that Reuters teams can implement to better position their content in the current industry context.
"""
        elif category == 'quantitative_analysis':
            return base_context + """
SPECIALIZED DEEP DIVE: QUANTITATIVE ANALYSIS

Perform a detailed quantitative analysis of the Teletrax data, focusing exclusively on:

1. Statistical Patterns and Correlations:
   - Calculate correlation coefficients between different variables (e.g., story type and detection length)
   - Identify statistically significant patterns in the data
   - Perform regression analysis where appropriate to identify predictive relationships
   - Test hypotheses about usage patterns with appropriate statistical methods

2. Trend Analysis:
   - Calculate percentage changes over time for key metrics
   - Identify acceleration or deceleration in usage patterns
   - Perform time series analysis to identify cyclical patterns
   - Quantify the strength and reliability of observed trends

3. Segmentation Analysis:
   - Segment the data in multiple meaningful ways (by time period, content type, etc.)
   - Calculate key metrics for each segment
   - Identify statistically significant differences between segments
   - Recommend optimal segmentation approaches for ongoing analysis

4. Outlier and Anomaly Detection:
   - Use statistical methods to identify outliers in the data
   - Analyze the nature and potential causes of these outliers
   - Determine if outliers represent opportunities or concerns
   - Suggest how to monitor for similar anomalies in the future

5. Predictive Modeling:
   - Develop simple predictive models based on the data
   - Estimate future usage patterns based on historical data
   - Identify key variables that drive usage patterns
   - Suggest how these models could be refined with additional data

Present your analysis with appropriate statistical terminology and visual representations (described in text). Include specific numbers, percentages, and statistical measures throughout. Your response should be data-driven and technically sound, while still being accessible to business stakeholders.
"""
        elif category == 'temporal_trends':
            return base_context + """
SPECIALIZED DEEP DIVE: TEMPORAL TRENDS ANALYSIS

Analyze how patterns in the Teletrax data change over time, focusing exclusively on:

1. Evolution of Preferences:
   - Analyze how client content preferences have evolved over the time period
   - Identify shifts in topic interest, content format, or usage patterns
   - Quantify the rate and direction of change for key metrics
   - Suggest potential causes for observed changes

2. Cyclical Patterns:
   - Identify daily, weekly, monthly, or seasonal patterns in content usage
   - Analyze the timing of peak usage periods and potential drivers
   - Determine if certain content types have specific temporal patterns
   - Recommend how Reuters can optimize content delivery based on these cycles

3. Event-Driven Changes:
   - Identify significant events that coincide with changes in usage patterns
   - Analyze the duration and magnitude of event-driven effects
   - Determine if usage returns to baseline after events or establishes new patterns
   - Suggest strategies for anticipating and responding to future events

4. Long-term Trends:
   - Distinguish between short-term fluctuations and long-term trends
   - Project how current trends might continue into the future
   - Identify potential inflection points where trends might change
   - Recommend how Reuters can position itself for anticipated future patterns

5. Comparative Time Period Analysis:
   - Compare usage patterns across different time periods
   - Identify consistent patterns vs. time-specific anomalies
   - Analyze how external factors in different time periods affected usage
   - Suggest how insights from past periods can inform future strategy

Provide extremely detailed analysis with specific examples from the data, including precise dates, time periods, and quantitative measures of change over time. Your response should be comprehensive and actionable, with concrete recommendations for how Reuters can leverage temporal insights to better serve this client.
"""
        elif category == 'recommendation_details':
            return base_context + """
SPECIALIZED DEEP DIVE: DETAILED RECOMMENDATIONS

Develop extremely detailed, actionable recommendations based on the Teletrax data, focusing exclusively on:

1. Implementation Planning:
   - Provide step-by-step implementation plans for each recommendation
   - Identify specific teams or roles responsible for each step
   - Establish realistic timelines for implementation phases
   - Anticipate potential implementation challenges and how to address them
   - Suggest pilot approaches before full-scale implementation

2. Expected Outcomes:
   - Detail specific, measurable outcomes for each recommendation
   - Establish baseline metrics and target improvements
   - Estimate timeframes for when results should become apparent
   - Identify leading indicators that would signal early success
   - Suggest methods for isolating the impact of specific recommendations

3. Success Metrics:
   - Define precise KPIs for measuring the success of each recommendation
   - Establish measurement methodologies and data collection processes
   - Recommend appropriate reporting cadences and formats
   - Suggest benchmarks for evaluating performance
   - Provide guidance on interpreting results and making adjustments

4. Prioritization Framework:
   - Assess each recommendation's potential impact and implementation difficulty
   - Create a prioritization matrix with clear criteria
   - Suggest logical sequencing of recommendations
   - Identify quick wins vs. long-term strategic initiatives
   - Recommend resource allocation across the recommendation portfolio

5. Detailed Next Steps:
   - Outline immediate actions required to begin implementation
   - Identify stakeholders who need to be engaged
   - Suggest specific meetings, deliverables, and decision points
   - Recommend communication strategies for the implementation process
   - Provide templates or frameworks for implementation planning

Provide extremely detailed, practical guidance that goes far beyond high-level recommendations. Your response should serve as an implementation playbook that Reuters teams can follow immediately to drive measurable improvements in client relationships and content strategy.
"""
        elif category == 'editorial_insights':
            return base_context + """
SPECIALIZED DEEP DIVE: EDITORIAL INSIGHTS ANALYSIS

Analyze the Teletrax data from an editorial perspective, focusing exclusively on insights for producers and output editors. Focus on:

1. Content Performance Analysis:
   - Analyze which types of stories receive the most consistent usage
   - Identify patterns in story selection that indicate editorial preferences
   - Determine if certain story formats (live, packaged, etc.) perform better
   - Compare performance across different news categories and topics

2. Editorial Decision Support:
   - Provide data-driven insights to guide editorial decision-making
   - Identify underserved content areas with potential for increased coverage
   - Suggest optimal timing for different types of content
   - Recommend story formats and approaches based on demonstrated preferences

3. Resource Allocation Guidance:
   - Analyze which coverage areas justify additional resources
   - Identify potential efficiencies in content production
   - Suggest optimal crew and equipment allocation based on content performance
   - Recommend coverage priorities based on client usage patterns

4. Programming Recommendations:
   - Develop specific programming suggestions based on usage patterns
   - Identify optimal story sequencing and packaging approaches
   - Suggest content themes that could be developed into series or special coverage
   - Recommend editorial calendar adjustments based on cyclical usage patterns

5. Editorial Quality Metrics:
   - Suggest metrics to evaluate editorial content performance
   - Establish benchmarks for different types of content
   - Recommend a framework for editorial performance reviews
   - Propose methods to test and refine editorial approaches

Provide extremely detailed analysis with specific examples from the data. Your response should be comprehensive and actionable, with concrete recommendations that Reuters editorial teams can implement immediately to optimize content creation and distribution.
"""
        elif category == 'marketing_insights':
            return base_context + """
SPECIALIZED DEEP DIVE: MARKETING INSIGHTS ANALYSIS

Analyze the Teletrax data from a marketing and business development perspective, focusing exclusively on insights for client-facing teams. Focus on:

1. Client Value Proposition:
   - Analyze how the client's usage patterns demonstrate Reuters' value
   - Identify specific content areas where Reuters provides unique value
   - Develop talking points for client conversations based on usage data
   - Suggest ways to articulate Reuters' differentiation based on demonstrated preferences

2. Relationship Development Strategies:
   - Identify opportunities to deepen the client relationship
   - Suggest specific touchpoints and engagement strategies
   - Recommend content showcases or presentations based on client interests
   - Develop a relationship roadmap with key milestones and objectives

3. Contract Renewal Support:
   - Analyze usage patterns to support contract renewal discussions
   - Identify content areas to highlight during renewal negotiations
   - Suggest potential new services or offerings based on usage patterns
   - Recommend pricing and packaging approaches based on demonstrated value

4. Cross-Selling Opportunities:
   - Identify potential for additional Reuters services based on usage patterns
   - Suggest complementary offerings that align with demonstrated preferences
   - Develop targeted pitches for specific additional services
   - Recommend bundling strategies for enhanced client value

5. Client Success Metrics:
   - Define what success looks like for this specific client relationship
   - Establish KPIs to track relationship health and growth
   - Suggest a framework for regular client business reviews
   - Recommend methods to demonstrate Reuters' value contribution

Provide extremely detailed analysis with specific examples from the data. Your response should be comprehensive and actionable, with concrete recommendations that Reuters marketing and client-facing teams can implement immediately to strengthen client relationships and drive business growth.
"""
        else:
            # Default prompt for unknown categories
            return base_context + f"""
SPECIALIZED DEEP DIVE: {category.upper().replace('_', ' ')} ANALYSIS

Analyze the Teletrax data with a focus on {category.replace('_', ' ')}, providing detailed insights and actionable recommendations. Consider:

1. Key Patterns and Trends:
   - Identify the most significant patterns related to {category.replace('_', ' ')}
   - Analyze how these patterns compare to expected norms
   - Determine if there are any surprising or counterintuitive findings
   - Suggest potential explanations for the observed patterns

2. Implications for Reuters:
   - Analyze what these patterns mean for Reuters' relationship with this client
   - Identify opportunities to better serve the client based on these insights
   - Suggest potential adjustments to Reuters' approach based on the data
   - Recommend ways to leverage these insights for mutual benefit

3. Actionable Recommendations:
   - Provide specific, detailed recommendations based on your analysis
   - Include implementation steps for each recommendation
   - Suggest metrics to track the success of these recommendations
   - Prioritize recommendations based on potential impact and feasibility

4. Future Considerations:
   - Predict how patterns might evolve in the future
   - Identify potential risks or challenges to address
   - Suggest proactive measures to stay ahead of changing needs
   - Recommend a framework for ongoing analysis and adjustment

5. Comparative Context:
   - Compare these patterns to what might be expected in the industry
   - Analyze how these insights relate to broader trends
   - Suggest how Reuters can position itself in light of these insights
   - Recommend ways to communicate these insights to relevant stakeholders

Provide extremely detailed analysis with specific examples from the data. Your response should be comprehensive and actionable, with concrete recommendations that Reuters teams can implement immediately.
"""
