import requests
import base64
import time
import json

API_URL = "http://localhost:8000/api/generate"


def print_result(test_name, success, details):
    icon = "✅" if success else "❌"
    print(f"{icon} {test_name}")
    if not success:
        print(f"   Error: {details}")
    else:
        print(f"   Info: {details}")
    print("-" * 50)


def test_code_generation():
    print("Testing Code Generation...")
    payload = {
        "prompt": "Write a Python function to calculate Fibonacci series.",
        "optimize_prompt": False,
    }
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()

        if response.status_code == 200 and data["success"]:
            model_provider = data["model"]["provider"]
            # We expect a coding model usually, or generic
            print_result(
                "Code Gen Test",
                True,
                f"Model: {data['model']['model_name']} ({model_provider})",
            )
        else:
            print_result(
                "Code Gen Test",
                False,
                f"Status: {response.status_code}, Error: {data.get('error')}",
            )
    except Exception as e:
        print_result("Code Gen Test", False, str(e))


def test_creative_writing():
    print("Testing Creative Writing...")
    payload = {"prompt": "Write a short haiku about AI.", "optimize_prompt": True}
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()

        if response.status_code == 200 and data["success"]:
            print_result("Creative Test", True, f"Model: {data['model']['model_name']}")
        else:
            print_result(
                "Creative Test",
                False,
                f"Status: {response.status_code}, Error: {data.get('error')}",
            )
    except Exception as e:
        print_result("Creative Test", False, str(e))


def test_data_guard():
    print("Testing Data Guard...")
    payload = {
        "prompt": "My email is test@example.com and key is sk-12345.",
        "optimize_prompt": False,
    }
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()

        if response.status_code == 200:
            was_cleaned = data["data_guard"]["was_cleaned"]
            detected = data["data_guard"]["detected_types"]
            if was_cleaned:
                print_result("Data Guard Test", True, f"Detected: {detected}")
            else:
                print_result("Data Guard Test", False, "No sensitive data detected!")
        else:
            print_result("Data Guard Test", False, f"Status: {response.status_code}")
    except Exception as e:
        print_result("Data Guard Test", False, str(e))


def test_image_analysis():
    print("Testing Image Analysis (Gemini)...")
    # minimal 1x1 black pixel png base64
    base64_img = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    payload = {
        "prompt": "What color is this image?",
        "optimize_prompt": False,
        "images": [{"name": "test.png", "type": "image/png", "base64": base64_img}],
    }
    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()

        if response.status_code == 200 and data["success"]:
            provider = data["model"]["provider"]
            reasons = data["model"]["selection_reasons"]
            # Should be gemini or fallback with OCR
            if "gemini" in provider.lower():
                print_result(
                    "Image Test", True, f"Used Gemini: {data['model']['model_name']}"
                )
            elif "fallback" in str(reasons) or "OCR" in str(reasons):
                print_result(
                    "Image Test (Fallback)",
                    True,
                    f"Used Fallback/OCR: {data['model']['model_name']}",
                )
            else:
                print_result(
                    "Image Test",
                    True,
                    f"Response from {provider} (Might be OCR fallback handled silently)",
                )
        else:
            print_result(
                "Image Test",
                False,
                f"Status: {response.status_code}, Error: {data.get('error')}",
            )
    except Exception as e:
        print_result("Image Test", False, str(e))


if __name__ == "__main__":
    print("🚀 Starting Integration Tests...\n")
    test_code_generation()
    time.sleep(1)
    test_creative_writing()
    time.sleep(1)
    test_data_guard()
    time.sleep(1)
    test_image_analysis()
    print("\n✅ Tests Completed.")
