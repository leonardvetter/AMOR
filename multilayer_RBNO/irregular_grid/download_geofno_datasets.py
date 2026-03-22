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
        target_dir = '/users/PAS3244/dvoderhol/data/Geo-FNO/data'
    
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
    
    diffeo_dir = os.path.join(data_dir, 'diffeo')
    meshes_dir = os.path.join(data_dir, 'meshes')
    omesh_dir = os.path.join(data_dir, 'omesh')
    rmesh_dir = os.path.join(data_dir, 'rmesh')
    interp_dir = os.path.join(data_dir, 'interp')
    
    elasticity_dirs = []
    if os.path.exists(diffeo_dir):
        elasticity_dirs.append(('diffeo', diffeo_dir))
    if os.path.exists(meshes_dir):
        elasticity_dirs.append(('meshes', meshes_dir))
    if os.path.exists(omesh_dir):
        elasticity_dirs.append(('omesh', omesh_dir))
    if os.path.exists(rmesh_dir):
        elasticity_dirs.append(('rmesh', rmesh_dir))
    if os.path.exists(interp_dir):
        elasticity_dirs.append(('interp', interp_dir))
    
    for dir_name, dir_path in elasticity_dirs:
        rr_files = [f for f in os.listdir(dir_path) if f.startswith('Random_UnitCell_rr') and f.endswith('.npy') and 'Deform' not in f and 'Grid' not in f]
        xy_files = [f for f in os.listdir(dir_path) if f.startswith('Random_UnitCell_XY') and f.endswith('.npy') and 'Deform' not in f and 'Grid' not in f]
        sigma_files = [f for f in os.listdir(dir_path) if f.startswith('Random_UnitCell_sigma') and f.endswith('.npy') and 'Deform' not in f and 'Grid' not in f]
        
        if rr_files and xy_files and sigma_files:
            datasets['elasticity'] = {
                'type': 'elasticity',
                'data_dir': dir_path,
                'input': os.path.join(dir_path, rr_files[0]),
                'mesh_coords': os.path.join(dir_path, xy_files[0]),
                'output': os.path.join(dir_path, sigma_files[0])
            }
            print(f"Found elasticity dataset in {dir_name}/ ({dir_path})")
            if len(rr_files) > 1:
                print(f"  Note: Multiple versions found. Available: {rr_files}")
            break
    
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
            print("  Expected files (base Geo-FNO model needs these exact names):")
            print("    - Random_UnitCell_rr_10.npy")
            print("    - Random_UnitCell_XY_10.npy")
            print("    - Random_UnitCell_sigma_10.npy")
            print("  These files should be in one of these folders:")
            print("    - data/diffeo/ (preferred)")
            print("    - data/meshes/")
            print("    - data/omesh/")
            print("    - data/rmesh/")
            print("    - data/interp/")
            print("  Note: Other variants (Deform, Grid, interp) are for different model types.")
            print("  Download from: https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
            print("  Check which folder contains the base files (without 'Deform' or 'Grid' in the name).")
        
        if not datasets:
            print("\nNo datasets found. Datasets need to be downloaded from Google Drive:")
            print("  https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8?usp=sharing")
            print(f"\nAfter downloading, extract to: {repo_path}/data/")
            print("  - Airfoil data should go in: data/naca/")
            print("  - Elasticity data: Check which Google Drive folder contains:")
            print("      Random_UnitCell_rr_10.npy, Random_UnitCell_XY_10.npy, Random_UnitCell_sigma_10.npy")
            print("    Then extract that folder's contents to: data/diffeo/ (or create symlink)")
        
        return datasets
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    datasets = main()
    sys.exit(0 if datasets else 1)

