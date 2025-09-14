"""
Complete training pipeline for the recommendation system
Run this after installing Python to train on real MovieLens data
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask', 'flask_cors', 'pandas', 'numpy', 
        'scikit-learn', 'scipy', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n🔧 Installing missing packages...")
        os.system(f"pip install {' '.join(missing_packages)}")
        print("✅ Installation completed!")
    else:
        print("✅ All required packages are installed!")

def main():
    """Main training pipeline"""
    print("🎬 Movie Recommendation System - Complete Training Pipeline")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    print("\n📊 Training on Real MovieLens Dataset...")
    print("-" * 40)
    
    try:
        # Import after ensuring dependencies are installed
        from train_model import main as train_main
        
        # Run training
        train_main()
        
        print("\n🎉 Training completed successfully!")
        print("\n📁 Files created:")
        print("   - model_results.json (training results)")
        print("   - data/movielens/ (downloaded dataset)")
        
        print("\n🚀 Next steps:")
        print("   1. Run: python app_with_training.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Click 'Train on Real Data' button")
        print("   4. Explore recommendations!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check internet connection (needs to download dataset)")
        print("   2. Ensure Python 3.7+ is installed")
        print("   3. Try: pip install --upgrade pip")
        print("   4. Try: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
