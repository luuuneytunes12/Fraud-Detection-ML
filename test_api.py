#!/usr/bin/env python3
"""
Simple test script to verify the fraud detection API works correctly.
"""

import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_prediction():
    """Test the prediction functionality directly."""
    try:
        from app.predict import make_prediction
        from app.main import InputFeatures

        # Create test data using the example from the Pydantic model
        test_data = InputFeatures(
            Time=0.0,
            V1=-1.359807,
            V2=-0.072781,
            V3=2.536347,
            V4=1.378155,
            V5=-0.338321,
            V6=0.462388,
            V7=0.239599,
            V8=0.098698,
            V9=0.363787,
            V10=0.090794,
            V11=-0.551600,
            V12=-0.617801,
            V13=-0.991390,
            V14=-0.311169,
            V15=1.468177,
            V16=-0.470401,
            V17=0.207971,
            V18=0.025791,
            V19=0.403993,
            V20=0.251412,
            V21=-0.018307,
            V22=0.277838,
            V23=-0.110474,
            V24=0.066928,
            V25=0.128539,
            V26=-0.189115,
            V27=0.133558,
            V28=-0.021053,
            Amount=149.62,
        )

        # Make prediction
        result = make_prediction(test_data)

        print("✅ Prediction test passed!")
        print(f"📊 Result: {json.dumps(result, indent=2)}")

        # Validate result structure
        assert "prediction" in result, "Result should contain 'prediction' key"
        assert "fraud_probability" in result, (
            "Result should contain 'fraud_probability' key"
        )
        assert "confidence" in result, "Result should contain 'confidence' key"

        print("✅ All validation checks passed!")
        return True

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_fastapi_import():
    """Test that FastAPI app can be imported."""
    try:
        from app.main import app

        print("✅ FastAPI app import test passed!")
        return True
    except Exception as e:
        print(f"❌ FastAPI import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing Fraud Detection API...")
    print("=" * 50)

    success = True

    # Test 1: FastAPI import
    print("\n1. Testing FastAPI app import...")
    success &= test_fastapi_import()

    # Test 2: Prediction functionality
    print("\n2. Testing prediction functionality...")
    success &= test_prediction()

    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The API is working correctly.")
        print("\n📝 To run the FastAPI server:")
        print("   cd /Users/lunlun/Downloads/Github/Fraud-Detection-ML")
        print("   source .venv/bin/activate")
        print(
            "   uvicorn app.fastapi_endpoints:app --host 0.0.0.0 --port 8080 --reload"
        )
    else:
        print("💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
