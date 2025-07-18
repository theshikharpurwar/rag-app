#!/usr/bin/env python3
"""
Simple script to test the upload functionality
"""
import requests
import os

def test_upload():
    # Test file path
    test_file = "python/test_files/sample.pdf"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    # Upload URL
    url = "http://localhost:5000/api/upload"
    
    try:
        with open(test_file, 'rb') as f:
            files = {'file': ('sample.pdf', f, 'application/pdf')}
            response = requests.post(url, files=files, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Upload successful!")
        else:
            print("❌ Upload failed!")
            
    except Exception as e:
        print(f"❌ Error during upload: {e}")

if __name__ == "__main__":
    test_upload()
