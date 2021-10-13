import os
import time
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ..helpers._utils import multiprocess_map
from .._logger import logger

__all__ = ["NumpyWriter"]


class NumpyWriter:
    """Write dataset into a set of numpy shards.

    Args:
        shard_size (int, optional):
            Number of images in one shard. Defaults to 1024.
        processes (int, optional):
            Number of saving processes. If undefined, will use 
            os.cpu_count() - 1 Defaults to None.
    """

    def __init__(self, shard_size: int = 1024, processes: int = None):
        self.shard_size = shard_size
        self.processes = processes

    def write_shards(
        self,
        output_dir: str,
        paths: List[str],
        labels: Dict[str, Dict[str, Any]] = None,
        pattern: str = "shard_%06d",
        max_samples: int = None,
        overwrite: bool = False,
    ):
        """Write a dataset as numpy shards.

        Args:
            output_dir (str):
                Where to save shards.
            paths (List[str]):
                List of paths to save.
            labels (Dict[str, Dict[str, Any]], optional):
                Dictonary of label dictionaries. Defaults to None. Format as:
                    labels = {
                        './path1': {'cancer': 1, 'death': 0, 'dog': 1}
                        './path2': {'cancer': 0, 'death': 0, 'dog': 0}
                        ...
                    }
            pattern (str, optional):
                Pattern for shards. Defaults to "shard_%06d".
            max_samples (int, optional):
                Maximum number of paths to save. Defaults to None.
            overwrite (bool, optional):
                Overwrite all shards matching pattern. Defaults to False.
        """
        if isinstance(labels, list):
            if len(labels) != len(paths):
                raise ValueError("Label and path lengths do not match!")
            else:
                labels = dict(zip(paths, [{"label": x} for x in labels]))
        try:
            self._write_shards(
                output_dir, paths, labels, pattern, max_samples, overwrite)
        except KeyboardInterrupt:
            logger.warning(
                "Keyboard interruption detected! You might want to check "
                "the last shard..."
            )

    def _write_shards(self, output_dir: str, paths: List[str],
                      labels: Dict[str, list], pattern: str, max_samples: int,
                      overwrite: bool):
        if "%" not in pattern:
            raise ValueError(
                "Pattern must contain '%' for numbering (eg. shard_%06d).")
        if (
            labels is not None and ( not (
                isinstance(labels, dict) and 
                isinstance(labels[next(iter(labels))], dict)
            ))
        ):
            raise ValueError(
                "Lables should be given as a nested dictionary. For example:"
                "    {path_1: {'cancer': 1, 'cohort': 2}, ...}"
            )
        # Check output_dir.
        full_pattern = os.path.join(output_dir, pattern)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        elif os.path.exists(full_pattern % 1 + ".npy"):
            if not overwrite:
                logger.warning(
                    f"Found shards matching {os.path.dirname(full_pattern)}. "
                    "Please set overwrite=True, change pattern or output_dir."
                )
                return
            else:
                logger.warning(f"Removing all shards matching {full_pattern}.")
                match = pattern.split("%")[0]
                rm = [f.path for f in os.scandir(
                    output_dir) if match in f.path]
                multiprocess_map(os.remove, rm)
        # Start timer.
        tic = time.time()
        # Generate chunk iterator from paths.
        path_chunks = chunks(paths, self.shard_size)
        # Check max shards and generate full path pattern.
        if max_samples is None:
            max_samples = len(paths)
        max_shards = max_samples // self.shard_size
        # Log log log.
        logger.info(f"Generating (approx.) {max_shards} shards "
                    f"with {self.shard_size} samples in each npy-file.")
        # Init while loop variables.
        shard_idx = 1
        shard_times = []
        queue = []
        while True:
            shard_tic = time.time()
            output_path = full_pattern % shard_idx
            prefix = f"[%{len(str(max_shards))}d/%d]" % (shard_idx, max_shards)
            logger.info(f"{prefix} Writing {pattern % shard_idx}. "
                        f"{self._get_etc(shard_times, max_shards - shard_idx)}")
            # Fill queue.
            queue = self._fill_queue(queue, path_chunks, labels)
            # Stop if queue has less samples than shard_size.
            if len(queue) < self.shard_size:
                break
            # Write to shard.
            queue = self._write_shard(output_path, queue)
            # Log shard time.
            shard_times.append(time.time() - shard_tic)
            # Stop if we reached max_shards.
            if max_shards == shard_idx:
                break
            # Increase shard_idx.
            shard_idx += 1

        s = time.time() - tic
        logger.info(
            f"Saved {shard_idx} shards to {output_dir} in {format_seconds(s)} "
            f"({(self.shard_size*shard_idx)/s:.2f} img/s)."
        )

    def _get_etc(self, times: list, num_left: int):
        if len(times) == 0:
            return ""
        else:
            return f"ETC: {format_seconds(np.mean(times)*num_left)}"

    def _fill_queue(self, queue, path_chunks, labels):
        """Fills queue until we have at least one chunk ready."""
        while len(queue) < self.shard_size:
            # Fill queues have at least shard_size samples.
            try:
                path_chunk = next(path_chunks)
            except StopIteration:
                break
            # Load images (parallel).
            image_chunk = multiprocess_map(
                func=_read_image, 
                lst=path_chunk, 
                processes=self.processes
            )
            image_chunk = [x for x in image_chunk if x is not None]
            for path, image in image_chunk:
                sample = {"labels": {"path": path}, "image": image}
                if (labels is not None and labels.get(path) is not None):
                    sample["labels"].update(labels[path])
                queue.append(sample)
        return queue

    def _write_shard(self, output_path, queue):
        # Pull chunk from queue.
        samples = queue[: self.shard_size]
        # Split into labels and images.
        labels = [x["labels"] for x in samples]
        images = [x["image"] for x in samples]
        # Write images to a npy-file.
        np.save(output_path + ".npy", np.array(images, dtype=object))
        # Write labels to a csv-file.
        pd.DataFrame(labels).to_csv(output_path + ".csv", index=False)
        # Return queue.
        return queue[self.shard_size:]


def format_seconds(n: int) -> str:
    """Format seconds into pretty string format."""
    days = int(n // (24 * 3600))
    n = n % (24 * 3600)
    hours = int(n // 3600)
    n %= 3600
    minutes = int(n // 60)
    n %= 60
    seconds = n
    if days > 0:
        strtime = f'{days}d {(hours)}h:{minutes}m:{int(seconds)}s'
    elif hours > 0:
        strtime = f'{(hours)}h:{minutes}m:{int(seconds)}s'
    elif minutes > 0:
        strtime = f'{minutes}m:{int(seconds)}s'
    else:
        strtime = f'{seconds:.2f}s'
    return strtime


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _read_image(path):
    """Read raw image from disk."""
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        image_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    return path, image_bytes