#!/usr/bin/env python3
"""
Test the fraud detection API with curl commands.
"""

import subprocess
import time
import json
import sys
import os


def test_api():
    """Test the API with curl commands."""

    print("🧪 Testing Fraud Detection API with curl...")
    print("=" * 60)

    # Test data for fraud prediction
    test_data = {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62,
    }

    base_url = "http://localhost:8082"

    # Test 1: Welcome endpoint
    print("\n1. Testing Welcome Endpoint...")
    print(f'📡 curl -X GET "{base_url}/predict/"')

    try:
        result = subprocess.run(
            ["curl", "-X", "GET", f"{base_url}/predict/"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"✅ Response: {result.stdout}")
        else:
            print(f"❌ Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Prediction endpoint
    print(f"\n2. Testing Prediction Endpoint...")
    print(f'📡 curl -X POST "{base_url}/predict" \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f"     -d '{json.dumps(test_data)}'")

    try:
        result = subprocess.run(
            [
                "curl",
                "-X",
                "POST",
                f"{base_url}/predict",
                "-H",
                "Content-Type: application/json",
                "-d",
                json.dumps(test_data),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"✅ Response: {result.stdout}")

            # Try to parse and pretty print the JSON response
            try:
                response_data = json.loads(result.stdout)
                print(f"📊 Parsed Response:")
                print(json.dumps(response_data, indent=2))
            except json.JSONDecodeError:
                print(f"📝 Raw Response: {result.stdout}")
        else:
            print(f"❌ Error: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("🏁 Test completed!")

    # Print curl commands for manual testing
    print("\n📋 Manual curl commands you can run:")
    print(f'   curl -X GET "{base_url}/predict/"')
    print(f'   curl -X POST "{base_url}/predict" \\')
    print(f'        -H "Content-Type: application/json" \\')
    print(f"        -d '{json.dumps(test_data, separators=(',', ':'))}'")


if __name__ == "__main__":
    test_api()
