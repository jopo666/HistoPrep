import os

import pandas as pd
from tqdm import tqdm

__all__ = [
    'combine_metadata'
]

def combine_metadata(parent_dir: str, save_csv: bool = False) -> pd.DataFrame:
    """Combine all metadata into a single csv-file."""
    dataframes = []
    directories = len([x.path for x in os.scandir(parent_dir)])
    for directory in tqdm(
            directories,
            total=len(directories),
            desc='Combining metadata',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            ):
        metadata_path = os.path.join(directory,'metadata.csv')
        # There might be empty files.
        if os.path.getsize(metadata_path) > 5:
            dataframes.append(pd.read_csv(f.path))
    metadata = pd.concat(dataframes)
    if save_csv:
        metadata.to_csv(os.path.join(parent_dir,'metadata.csv'), index=False)
    return metadata
