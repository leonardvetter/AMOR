"""
Script to download Geo-FNO datasets (airfoil and elasticity).

This script clones the Geo-FNO repository and locates the dataset files.
"""

import os
import subprocess
import sys
from pathlib import Path


def clone_geofno_repo(target_dir: str = None):
    """
    Clone Geo-FNO repository.
    
    Args:
        target_dir: Directory to clone into (default: $SCRATCH/data)
    
    Returns:
        Path to cloned repository
    """
    if target_dir is None:
        scratch_dir = os.environ.get('SCRATCH', '/pscratch/sd/d/dvod')
        target_dir = os.path.join(scratch_dir, 'data')
    
    os.makedirs(target_dir, exist_ok=True)
    repo_path = os.path.join(target_dir, 'Geo-FNO')
    
    if os.path.exists(repo_path):
        print(f"Repository already exists at {repo_path}")
        return repo_path
    
    print(f"Cloning Geo-FNO repository to {repo_path}...")
    
    repos = [
        'https://github.com/neuraloperator/Geo-FNO.git',
        'https://github.com/lu-group/Geo-FNO.git'
    ]
    
    for repo_url in repos:
        try:
            result = subprocess.run(
                ['git', 'clone', repo_url, repo_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print(f"Successfully cloned from {repo_url}")
                return repo_path
            else:
                print(f"Failed to clone from {repo_url}: {result.stderr}")
        except Exception as e:
            print(f"Error cloning from {repo_url}: {e}")
            continue
    
    raise RuntimeError("Failed to clone Geo-FNO repository from any source")


def find_geofno_datasets(repo_path: str):
    """
    Find airfoil and elasticity datasets in Geo-FNO repository.
    
    Geo-FNO datasets are stored as .npy files in data/ directory:
    - Airfoil: data/naca/NACA_Cylinder_X.npy, NACA_Cylinder_Y.npy, NACA_Cylinder_Q.npy
    - Elasticity: data/diffeo/Random_UnitCell_rr_10.npy, Random_UnitCell_XY_10.npy, Random_UnitCell_sigma_10.npy
    
    Args:
        repo_path: Path to Geo-FNO repository
        
    Returns:
        Dictionary with dataset information
    """
    datasets = {}
    
    data_dir = os.path.join(repo_path, 'data')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Datasets need to be downloaded from Google Drive:")
        print("https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
        return datasets
    
    naca_dir = os.path.join(data_dir, 'naca')
    diffeo_dir = os.path.join(data_dir, 'diffeo')
    
    if os.path.exists(naca_dir):
        x_file = os.path.join(naca_dir, 'NACA_Cylinder_X.npy')
        y_file = os.path.join(naca_dir, 'NACA_Cylinder_Y.npy')
        q_file = os.path.join(naca_dir, 'NACA_Cylinder_Q.npy')
        
        if all(os.path.exists(f) for f in [x_file, y_file, q_file]):
            datasets['airfoil'] = {
                'type': 'airfoil',
                'data_dir': naca_dir,
                'input_x': x_file,
                'input_y': y_file,
                'output': q_file
            }
            print(f"Found airfoil dataset in {naca_dir}")
    
    if os.path.exists(diffeo_dir):
        rr_file = os.path.join(diffeo_dir, 'Random_UnitCell_rr_10.npy')
        xy_file = os.path.join(diffeo_dir, 'Random_UnitCell_XY_10.npy')
        sigma_file = os.path.join(diffeo_dir, 'Random_UnitCell_sigma_10.npy')
        
        if all(os.path.exists(f) for f in [rr_file, xy_file, sigma_file]):
            datasets['elasticity'] = {
                'type': 'elasticity',
                'data_dir': diffeo_dir,
                'input': rr_file,
                'mesh_coords': xy_file,
                'output': sigma_file
            }
            print(f"Found elasticity dataset in {diffeo_dir}")
    
    return datasets


def main():
    """Main function to download and locate Geo-FNO datasets."""
    print("=" * 60)
    print("Geo-FNO Dataset Downloader")
    print("=" * 60)
    
    try:
        repo_path = clone_geofno_repo()
        datasets = find_geofno_datasets(repo_path)
        
        print("\n" + "=" * 60)
        print("Dataset Summary:")
        print("=" * 60)
        
        if 'airfoil' in datasets:
            print(f"Airfoil dataset found:")
            print(f"  Data directory: {datasets['airfoil']['data_dir']}")
            print(f"  Use this path with train_geofno.py:")
            print(f"    python train_geofno.py --dataset_path {datasets['airfoil']['data_dir']} --dataset_type airfoil")
        else:
            print("Airfoil dataset: Not found")
            print("  Expected files in data/naca/:")
            print("    - NACA_Cylinder_X.npy")
            print("    - NACA_Cylinder_Y.npy")
            print("    - NACA_Cylinder_Q.npy")
            print("  Download from: https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
        
        if 'elasticity' in datasets:
            print(f"Elasticity dataset found:")
            print(f"  Data directory: {datasets['elasticity']['data_dir']}")
            print(f"  Use this path with train_geofno.py:")
            print(f"    python train_geofno.py --dataset_path {datasets['elasticity']['data_dir']} --dataset_type elasticity")
        else:
            print("Elasticity dataset: Not found")
            print("  Expected files in data/diffeo/:")
            print("    - Random_UnitCell_rr_10.npy")
            print("    - Random_UnitCell_XY_10.npy")
            print("    - Random_UnitCell_sigma_10.npy")
            print("  Download from: https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
        
        if not datasets:
            print("\nNo datasets found. Datasets need to be downloaded from Google Drive:")
            print("  https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
            print(f"\nAfter downloading, extract to: {repo_path}/data/")
            print("  - Airfoil data should go in: data/naca/")
            print("  - Elasticity data should go in: data/diffeo/")
        
        return datasets
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    datasets = main()
    sys.exit(0 if datasets else 1)

