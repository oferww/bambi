#!/usr/bin/env python3
"""
Test script to verify Bambi setup (relocated to testing/ for local dev convenience)
"""

import os
import sys
from dotenv import load_dotenv


def test_environment():
    """Test environment variables and API key."""
    print("🔍 Testing environment...")

    load_dotenv()

    # Check Cohere API key
    api_key = os.getenv("COHERE_API_KEY_CHAT")
    if not api_key:
        print("❌ COHERE_API_KEY_CHAT not found in environment variables")
        print("   Please set it in your .env file")
        return False
    else:
        print("✅ COHERE_API_KEY_CHAT found")
        return True


def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")

    try:
        import streamlit  # noqa: F401
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

    try:
        import cohere  # noqa: F401
        print("✅ Cohere imported")
    except ImportError as e:
        print(f"❌ Cohere import failed: {e}")
        return False

    try:
        import chromadb  # noqa: F401
        print("✅ ChromaDB imported")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False

    try:
        from langchain_community.embeddings import CohereEmbeddings  # noqa: F401
        print("✅ LangChain Cohere embeddings imported")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False

    try:
        from PIL import Image  # noqa: F401
        print("✅ Pillow imported")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False

    try:
        import exifread  # noqa: F401
        print("✅ ExifRead imported")
    except ImportError as e:
        print(f"❌ ExifRead import failed: {e}")
        return False

    return True


def test_local_components():
    """Test local components."""
    print("🔍 Testing local components...")

    try:
        from backend.chatbot import OferGPT  # noqa: F401
        print("✅ Bambi chatbot imported")
    except ImportError as e:
        print(f"❌ Bambi chatbot import failed: {e}")
        return False

    try:
        from backend.rag_system import RAGSystem  # noqa: F401
        print("✅ RAG system imported")
    except ImportError as e:
        print(f"❌ RAG system import failed: {e}")
        return False

    try:
        from backend.utils.photo_processor import PhotoProcessor  # noqa: F401
        print("✅ Photo processor imported")
    except ImportError as e:
        print(f"❌ Photo processor import failed: {e}")
        return False

    return True


def test_directories():
    """Test that required directories exist."""
    print("🔍 Testing directories...")

    required_dirs = [
        "data",
        "data/photos",
        "data/embeddings",
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Directory not found: {dir_path}")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
            except Exception as e:
                print(f"❌ Failed to create directory {dir_path}: {e}")
                return False
        else:
            print(f"✅ Directory exists: {dir_path}")

    return True


def test_cohere_api():
    """Test Cohere API connection."""
    print("🔍 Testing Cohere API...")

    try:
        import cohere
        client = cohere.Client(os.getenv("COHERE_API_KEY_CHAT"))

        # Test with a simple request
        response = client.generate(
            model="command-a-vision-07-2025",
            prompt="Hello",
            max_tokens=10,
        )
        _ = response
        print("✅ Cohere API connection successful")
        return True
    except Exception as e:
        print(f"❌ Cohere API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Bambi setup tests...\n")

    tests = [
        ("Environment Variables", test_environment),
        ("Package Imports", test_imports),
        ("Local Components", test_local_components),
        ("Directories", test_directories),
        ("Cohere API", test_cohere_api),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)

    if passed == total:
        print("🎉 All tests passed! Your Bambi setup is ready.")
        print("\nNext steps:")
        print("1. Add photos to data/photos/")
        print("2. Run: docker-compose up --build")
        print("3. Open http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
