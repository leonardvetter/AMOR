#!/usr/bin/env python3
"""Download Geo-FNO datasets from Google Drive."""
import subprocess
import sys
import os
import site

# Add user site-packages to path
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

# Install dependencies first
print("Installing dependencies...")
try:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--user', 
        'gdown', 'beautifulsoup4', 'lxml'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    print(f"Warning: pip install had issues: {e}")

# Reload site to pick up new packages
import importlib
importlib.reload(site)

# Now import and use
try:
    import gdown
except ImportError:
    # Try adding common user site-packages locations
    for path in [
        '/global/homes/d/dvod/.local/perlmutter/pytorch2.8.0/lib/python3.12/site-packages',
        '/global/homes/d/dvod/.local/lib/python3.12/site-packages',
        os.path.expanduser('~/.local/lib/python3.12/site-packages'),
    ]:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    import gdown

data_dir = '/mnt/c/Users/dvod/MRBNO/data/Geo-FNO/data'

folders = [
    ('1PMuOGUO2A9rzAP3YM8ra3v8ybE8X9foh', 'car-cfd'),
    ('1LD6BLfdGcS0AY2TbjvPxarl7Bd6u9bY4', 'channel-shocks'),
    ('1iOgLyfqGtXuZ25KHsDWIh_ZJAfO_HxMO', 'pipe'),
    ('1lo7n8dWtpcVhhaEpQavh9UoSkMOs1UdC', 'plasticity'),
    ('1_pGrwS6rwzhyaCXS6MEmmPKWFXNeRnhj', 'naca'),
    ('1cznHmQO-hB_VlWOfh7IFpjsPVOYyghGJ', 'diffeo'),
]

print("\n" + "="*70)
print("Downloading all 6 Google Drive folders")
print("="*70)

for folder_id, target_name in folders:
    output_path = os.path.join(data_dir, target_name)
    os.makedirs(output_path, exist_ok=True)
    
    existing = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]) if os.path.exists(output_path) else 0
    
    print(f"\n[{target_name}] {existing} existing files")
    print(f"[{target_name}] Downloading folder {folder_id}...")
    
    try:
        gdown.download_folder(
            f'https://drive.google.com/drive/folders/{folder_id}',
            output=output_path,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        
        files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        size_mb = sum(os.path.getsize(os.path.join(output_path, f)) for f in files) / (1024**2)
        print(f"[{target_name}] ✓ {len(files)} files, {size_mb:.1f} MB")
        
    except Exception as e:
        print(f"[{target_name}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("Final Summary")
print("="*70)
for folder_id, target_name in folders:
    output_path = os.path.join(data_dir, target_name)
    if os.path.exists(output_path):
        files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
        if files:
            size_mb = sum(os.path.getsize(os.path.join(output_path, f)) for f in files) / (1024**2)
            print(f"{target_name}: {len(files)} files, {size_mb:.1f} MB")
