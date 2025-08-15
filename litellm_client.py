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
            teletrax_data: Dictionary containing processed Teletrax data
            channel_name: Name of the channel being analyzed
            
        Returns:
            Dictionary containing system and user prompts
        """
        # System prompt with instructions for the LLM
        system_prompt = """
You are a world-class data analytics expert, hired by Reuters to provide objective, balanced client insights based on story usage data. Your primary goal is to deliver an honest, data-driven analysis that identifies both positive and negative trends to help Reuters make informed decisions. When a user provides you with Teletrax data, be as detailed as possible in your analysis. Provide a comprehensive report that includes both strengths and areas for improvement. Max out your tokens. Find the most nuanced details you can, see the signal in the noise.

CRITICAL DATA CONTEXT - READ CAREFULLY:
The data you are analyzing represents ONLY the detections from the specific channels included in the upload. It does NOT represent all Reuters content production or global usage patterns. This means:
1. The stories, counts, and statistics in the data only show how these specific channels used Reuters content
2. Percentages and proportions are relative ONLY to the channels in this dataset, not to Reuters' global performance
3. You cannot make conclusions about Reuters' overall content strategy or production volume from this data
4. When you see "73% of detections" or similar statistics, this means "73% of the detections in this specific dataset"
5. The data does not show how many stories Reuters produced in total - only those that were detected on these specific channels

IMPORTANT: Your analysis MUST be balanced and critical:
1. Explicitly identify declining trends when present (e.g., decreasing usage over time, declining interest in certain content types)
2. Highlight potential concerns or issues that require attention
3. Avoid overly positive language that isn't supported by the data
4. Do not make assumptions about client satisfaction - stick to what the data shows
5. Be direct and honest about underperforming content or missed opportunities
6. Provide specific, actionable recommendations to address negative trends

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

Teletrax Data Analysis Prompt for Reuters:

Objective:
Analyze Teletrax data to generate insights on video performance, client engagement, and regional usage trends for Reuters. The output should be in clear, simple language that is easy for a producer to understand, avoiding unnecessary Teletrax jargon. The goal is to present actionable insights rather than raw data, ensuring the analysis is practical and relevant for editorial and commercial decisions.

Your analysis should be structured in the following way:
1. Executive Summary - A brief overview of the key findings
2. Insights by Audience:
   - For Journalists and Producers
   - For Output Editors
   - For Marketing and Client-Facing Teams
3. Insights by Type:
   - Content Strategy Insights
   - Client Engagement Insights
   - Market Positioning Insights
4. Actionable Recommendations
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
