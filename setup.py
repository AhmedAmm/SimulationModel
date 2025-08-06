#!/usr/bin/env python3
"""
Simplified installation script for Windows
"""

import subprocess
import sys
import platform

def run_pip_install(packages, description=""):
    """Install packages with pip"""
    if isinstance(packages, str):
        packages = [packages]
    
    for package in packages:
        print(f"🔧 Installing {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], check=True, capture_output=True, text=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"Error output: {e.stderr}")
            return False
    return True

def test_imports():
    """Test if packages can be imported"""
    test_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"), 
        ("scipy", "SciPy"),
        ("pyaudio", "PyAudio"),
        ("whisper", "OpenAI Whisper")
    ]
    
    print("\n🔍 Testing installations...")
    failed = []
    
    for module, name in test_packages:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: Failed - {e}")
            failed.append(name)
    
    return len(failed) == 0, failed

def main():
    print("🚀 Real-time Presentation System - Simple Installation")
    print("=" * 55)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Current: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Upgrade pip
    print("\n🔧 Upgrading pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                  capture_output=True)
    
    # Install packages step by step
    installation_steps = [
        (["numpy", "scipy"], "Core scientific packages"),
        (["requests"], "HTTP library"),
        (["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"], "PyTorch (CPU)"),
        (["pyaudio"], "Audio processing"),
        (["openai-whisper"], "OpenAI Whisper")
    ]
    
    print("\n📦 Installing packages...")
    for packages, description in installation_steps:
        print(f"\n🔧 Installing {description}...")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {description} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {description}")
            print(f"Command: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            
            # Special handling for PyAudio on Windows
            if "pyaudio" in packages:
                print("\n🔄 Trying alternative PyAudio installation...")
                try:
                    # Try pipwin approach
                    subprocess.run([sys.executable, "-m", "pip", "install", "pipwin"], 
                                 check=True, capture_output=True)
                    subprocess.run([sys.executable, "-m", "pipwin", "install", "pyaudio"], 
                                 check=True, capture_output=True)
                    print("✅ PyAudio installed via pipwin")
                except subprocess.CalledProcessError:
                    print("❌ Alternative PyAudio installation also failed")
                    print("📝 Manual PyAudio installation required:")
                    print("   1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
                    print("   2. Download appropriate wheel for your Python version")
                    print("   3. pip install downloaded_wheel.whl")
    
    # Test installations
    success, failed = test_imports()
    
    if success:
        print("\n🎉 All packages installed successfully!")
        print("\n📋 Next steps:")
        print("   1. Install FFmpeg:")
        print("      - Download from: https://ffmpeg.org/download.html")
        print("      - Add to system PATH")
        print("   2. Test the system:")
        print("      - python test_whisper.py")
        print("   3. Run full system:")
        print("      - python main.py")
    else:
        print(f"\n⚠️ Some packages failed to install: {', '.join(failed)}")
        print("Please install them manually before proceeding.")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        input("\nPress Enter to exit...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Installation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)