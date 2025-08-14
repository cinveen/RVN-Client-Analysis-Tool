"""
Load environment variables from .env file.

This module loads environment variables from a .env file into the environment.
It should be imported at the beginning of the application to ensure that
environment variables are available to all modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if required environment variables are set
if not os.environ.get('LITELLM_API_KEY'):
    print("Warning: LITELLM_API_KEY environment variable is not set.")
    print("Please create a .env file with your API key or set it in your environment.")
    print("Example .env file:")
    print("LITELLM_API_KEY=your_api_key_here")
    print("LITELLM_API_URL=https://litellm.int.thomsonreuters.com")
