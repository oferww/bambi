#!/usr/bin/env python3
"""
Setup script for Bambi (relocated to testing/ for local dev convenience)
"""

import os
import shutil
import subprocess
import sys

def create_env_file():
    """Create .env file from template."""
    if not os.path.exists('.env'):
        if os.path.exists('env.example'):
            shutil.copy('env.example', '.env')
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and add your Cohere API key")
        else:
            print("❌ env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'data/photos',
        'data/embeddings'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
            return True
        else:
            print("❌ Docker is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available."""
    try:
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose is available")
            return True
        else:
            print("❌ Docker Compose is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose is not installed")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Bambi...\n")
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check Docker
    docker_ok = check_docker()
    docker_compose_ok = check_docker_compose()
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    
    print("\nNext steps:")
    print("1. Edit .env file and add your Cohere API key")
    print("2. Add photos to data/photos/ directory")
    print("3. Run the application:")
    
    if docker_ok and docker_compose_ok:
        print("   docker-compose up --build")
    else:
        # Updated local dev guidance (split requirements)
        print("   pip install -r requirements-base.txt")
        print("   pip install -r requirements-app.txt")
        print("   streamlit run app.py")
    
    print("4. Open http://localhost:8501")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
