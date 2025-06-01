import os
import json
import requests
import pandas as pd
from pyinaturalist import get_observations, get_taxa
from tqdm import tqdm

def create_directory_structure():
    """Create required directory structure"""
    dirs = [
        'data/images',
        'models',
        'plots',
        'reports'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directory structure created")

def download_species_images(species_list, max_images=200):
    """Download images for each species from iNaturalist"""
    species_data = []
    
    for species in tqdm(species_list, desc="Downloading species images"):
        try:
            # Get taxon ID
            response = get_taxa(q=species, rank='species')
            if not response['results']:
                print(f"\nSpecies not found: {species}")
                continue
                
            taxon_id = response['results'][0]['id']
            species_dir = f"data/images/{species.replace(' ', '_')}"
            os.makedirs(species_dir, exist_ok=True)
            
            # Get observations
            observations = get_observations(
                taxon_id=taxon_id,
                quality_grade='research',
                per_page=max_images
            )['results']
            
            downloaded = 0
            for obs in observations:
                if downloaded >= max_images:
                    break
                    
                if obs.get('photos'):
                    url = obs['photos'][0]['url'].replace('square', 'medium')
                    try:
                        img_data = requests.get(url, timeout=10).content
                        with open(f"{species_dir}/{obs['id']}.jpg", 'wb') as f:
                            f.write(img_data)
                        downloaded += 1
                    except Exception as e:
                        continue
            
            species_data.append({
                "species": species,
                "taxon_id": taxon_id,
                "image_count": downloaded
            })
            print(f"\nDownloaded {downloaded} images for {species}")
            
        except Exception as e:
            print(f"\nError processing {species}: {str(e)}")
    
    # Save species metadata
    pd.DataFrame(species_data).to_csv('data/species_metadata.csv', index=False)
    return species_data
