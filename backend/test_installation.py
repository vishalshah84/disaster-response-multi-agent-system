#!/usr/bin/env python3
# Test script for installation verification
import sys

def test_imports():
    print("Testing imports...")
    try:
        import fastapi
        print("  ✅ FastAPI")
    except ImportError:
        print("  ❌ FastAPI not found")
        return False
    
    try:
        import uvicorn
        print("  ✅ Uvicorn")
    except ImportError:
        print("  ❌ Uvicorn not found")
        return False
    
    try:
        import numpy
        print("  ✅ NumPy")
    except ImportError:
        print("  ❌ NumPy not found")
        return False
    
    return True

def test_config():
    print("\nTesting configuration...")
    try:
        from app.core.config import settings, AgentType, BehaviorMode
        print(f"  ✅ Settings loaded: {settings.APP_NAME}")
        print(f"  ✅ Grid size: {settings.GRID_WIDTH}x{settings.GRID_HEIGHT}")
        print(f"  ✅ Agent types: {AgentType.FIRE_FIGHTER}, {AgentType.MEDICAL}")
        print(f"  ✅ Behavior modes: {BehaviorMode.COMPETITIVE}, {BehaviorMode.COOPERATIVE}")
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_app():
    print("\nTesting FastAPI app...")
    try:
        from app.main import app
        print("  ✅ FastAPI app loaded")
        return True
    except Exception as e:
        print(f"  ❌ App error: {e}")
        return False

if __name__ == "__main__":
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Installation Test - Disaster Response Simulation          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()
    print(f"Python version: {sys.version}")
    print()
    
    all_pass = True
    all_pass &= test_imports()
    all_pass &= test_config()
    all_pass &= test_app()
    
    print()
    print("═" * 60)
    if all_pass:
        print("✅ ALL TESTS PASSED!")
        print()
        print("🚀 Next steps:")
        print("   1. Install dependencies:")
        print("      pip3 install --break-system-packages -r requirements.txt")
        print()
        print("   2. Start the server:")
        print("      python3 -m uvicorn app.main:app --reload --port 8000")
        print()
        print("   3. Open browser:")
        print("      http://localhost:8000")
        print()
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Fix issues and run again:")
        print("  python3 test_installation.py")
    print("═" * 60)
