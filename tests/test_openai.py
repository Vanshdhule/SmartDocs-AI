# tests/test_api_helper.py
"""Manual smoke test for OpenAIHelper (now NVIDIA NIM). Run with: python tests/test_api_helper.py"""
from backend.openai_helper import OpenAIHelper

def test_nvidia_connection():
    try:
        helper = OpenAIHelper()
        print("API key loaded successfully")
        if helper.test_connection():
            print("NVIDIA NIM API connection successful!")
        response = helper.get_completion("Reply with just: Yes")
        print("\nNVIDIA NIM Response:", response)
    except Exception as e:
        print("\nERROR:", e)

if __name__ == "__main__":
    test_nvidia_connection()