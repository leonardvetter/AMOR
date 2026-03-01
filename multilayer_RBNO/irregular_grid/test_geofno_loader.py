"""
Test script to verify Geo-FNO dataset loading and download status.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=" * 60)
    print("Testing Geo-FNO Dataset Detection")
    print("=" * 60)
    
    from download_geofno_datasets import find_geofno_datasets, clone_geofno_repo
    
    repo_path = "/mnt/c/Users/dvod/MRBNO/data/Geo-FNO"
    
    if not os.path.exists(repo_path):
        print(f"Repository not found at {repo_path}")
        print("Attempting to clone...")
        try:
            repo_path = clone_geofno_repo()
        except Exception as e:
            print(f"Failed to clone: {e}")
            return 1
    
    print(f"\nChecking repository at: {repo_path}")
    datasets = find_geofno_datasets(repo_path)
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    
    if 'airfoil' in datasets:
        print(f"✓ Airfoil dataset found: {datasets['airfoil']['data_dir']}")
    else:
        print("✗ Airfoil dataset not found")
    
    if 'elasticity' in datasets:
        print(f"✓ Elasticity dataset found: {datasets['elasticity']['data_dir']}")
    else:
        print("✗ Elasticity dataset not found")
    
    if datasets:
        print("\n" + "=" * 60)
        print("Testing data loading...")
        print("=" * 60)
        
        try:
            from geofno_loader import load_geofno_dataset
            
            if 'elasticity' in datasets:
                print("\nTesting elasticity dataset loading...")
                data_dir = datasets['elasticity']['data_dir']
                data, targets, mesh_coords = load_geofno_dataset(data_dir, 'elasticity')
                print(f"  Input shape: {data.shape}")
                print(f"  Target shape: {targets.shape}")
                print(f"  Mesh coords shape: {mesh_coords.shape}")
                print("  ✓ Elasticity dataset loads successfully!")
            
            if 'airfoil' in datasets:
                print("\nTesting airfoil dataset loading...")
                data_dir = datasets['airfoil']['data_dir']
                data, targets, mesh_coords = load_geofno_dataset(data_dir, 'airfoil')
                print(f"  Input shape: {data.shape}")
                print(f"  Target shape: {targets.shape}")
                print(f"  Mesh coords shape: {mesh_coords.shape}")
                print("  ✓ Airfoil dataset loads successfully!")
                
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    return 0 if datasets else 1

if __name__ == "__main__":
    sys.exit(main())

