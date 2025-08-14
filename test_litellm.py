"""
Test script to verify that the LiteLLM client can connect to the API
using the environment variables loaded from .env file.
"""

import os
import json
from litellm_client import LiteLLMClient

# Load environment variables from .env file
import load_env

def main():
    # Check if the API key is loaded from environment variables
    api_key = os.environ.get('LITELLM_API_KEY')
    api_url = os.environ.get('LITELLM_API_URL')
    
    print(f"API URL: {api_url}")
    print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'Not found'}")
    
    # Create a simple test data structure
    test_data = {
        'stats': {
            'total_records': 100,
            'date_range': '2024-01-01 to 2025-07-01',
            'markets': 'US, UK, France',
            'unique_stories': 50,
            'unique_story_ids': 45
        },
        'top_stories': {
            'Story 1': 20,
            'Story 2': 15,
            'Story 3': 10
        },
        'detection_patterns': {
            'peak_hours': '8:00, 12:00, 18:00',
            'peak_days': 'Monday, Wednesday, Friday'
        },
        'country_distribution': {
            'US': 40,
            'UK': 30,
            'France': 20,
            'Germany': 10
        }
    }
    
    try:
        # Initialize the LiteLLM client
        client = LiteLLMClient()
        
        print("LiteLLM client initialized successfully")
        
        # Generate a test analysis
        print("Generating test analysis...")
        analysis = client.generate_analysis(test_data, "Test Channel")
        
        # Print the analysis structure (not the full content)
        print("\nAnalysis generated successfully!")
        print(f"Analysis timestamp: {analysis.get('timestamp')}")
        print(f"Analysis sections: {list(analysis.keys())}")
        
        # Check if the executive summary was generated
        exec_summary = analysis.get('executive_summary', '')
        if exec_summary:
            print(f"\nExecutive Summary preview: {exec_summary[:100]}...")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return False

if __name__ == "__main__":
    main()
