"""
Combine multiple speech emotion recognition datasets into a single unified structure.

Datasets supported:
- RAVDESS: Ryerson Audio-Visual Emotion Database
- TESS: Toronto Emotional Speech Set
- CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset

Target emotions: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
Mapping: calm → neutral, surprise → (discarded)

Output structure:
    combined_dataset/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    └── sad/
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# RAVDESS emotion mapping (filename field 2)
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'neutral',  # calm → neutral
    '03': 'happy',
    '04': 'sad', 
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': None,  # surprise → skip
}

# CREMA-D emotion mapping (filename codes)
CREMAD_EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad',
}

# TESS uses folder names directly (with some normalization)
TESS_EMOTION_MAP = {
    'angry': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'ps': None,  # pleasant surprise → skip
}


# ============================================================================
# DATASET PROCESSORS
# ============================================================================

class RavdessProcessor:
    """Process RAVDESS dataset."""
    
    @staticmethod
    def get_emotion(filename: str) -> Optional[str]:
        """
        Extract emotion from RAVDESS filename.
        Format: DD-TT-EE-AA-RR-SS.wav
        where EE is the emotion code (01-08)
        """
        try:
            parts = filename.replace('.wav', '').split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                return RAVDESS_EMOTION_MAP.get(emotion_code)
        except Exception as e:
            logger.warning(f"Failed to parse RAVDESS filename {filename}: {e}")
        return None
    
    @staticmethod
    def process(dataset_path: str, output_dir: str) -> Dict[str, int]:
        """Process RAVDESS dataset."""
        stats = {emotion: 0 for emotion in TARGET_EMOTIONS}
        stats['skipped'] = 0
        
        ravdess_path = Path(dataset_path) / 'ravdess'
        
        if not ravdess_path.exists():
            logger.warning(f"RAVDESS path not found: {ravdess_path}")
            return stats
        
        logger.info(f"Processing RAVDESS from {ravdess_path}")
        
        # Iterate through all actor folders
        for actor_folder in sorted(ravdess_path.iterdir()):
            if not actor_folder.is_dir():
                continue
            
            for wav_file in actor_folder.glob('*.wav'):
                emotion = RavdessProcessor.get_emotion(wav_file.name)
                
                if emotion is None:
                    stats['skipped'] += 1
                    continue
                
                # Create new filename: ravdess_<original>.wav
                new_name = f"ravdess_{wav_file.name}"
                dest_path = Path(output_dir) / emotion / new_name
                
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wav_file, dest_path)
                    stats[emotion] += 1
                except Exception as e:
                    logger.error(f"Failed to copy {wav_file} to {dest_path}: {e}")
        
        return stats


class TessProcessor:
    """Process TESS dataset."""
    
    @staticmethod
    def get_emotion(folder_name: str) -> Optional[str]:
        """
        Extract emotion from TESS folder name.
        TESS uses folder structure: <emotion>/ containing audio files
        """
        emotion = folder_name.lower().strip()
        return TESS_EMOTION_MAP.get(emotion)
    
    @staticmethod
    def process(dataset_path: str, output_dir: str) -> Dict[str, int]:
        """Process TESS dataset."""
        stats = {emotion: 0 for emotion in TARGET_EMOTIONS}
        stats['skipped'] = 0
        
        tess_path = Path(dataset_path) / 'TESS'
        
        if not tess_path.exists():
            logger.warning(f"TESS path not found: {tess_path}")
            return stats
        
        logger.info(f"Processing TESS from {tess_path}")
        
        # Iterate through emotion folders
        for emotion_folder in tess_path.iterdir():
            if not emotion_folder.is_dir():
                continue
            
            emotion = TessProcessor.get_emotion(emotion_folder.name)
            
            if emotion is None:
                stats['skipped'] += len(list(emotion_folder.glob('*.wav')))
                continue
            
            # Copy all audio files from this emotion folder
            for wav_file in emotion_folder.glob('*.wav'):
                # Create new filename: tess_<original>.wav
                new_name = f"tess_{wav_file.name}"
                dest_path = Path(output_dir) / emotion / new_name
                
                try:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(wav_file, dest_path)
                    stats[emotion] += 1
                except Exception as e:
                    logger.error(f"Failed to copy {wav_file} to {dest_path}: {e}")
        
        return stats


class CremadProcessor:
    """Process CREMA-D dataset."""
    
    @staticmethod
    def get_emotion(filename: str) -> Optional[str]:
        """
        Extract emotion from CREMA-D filename.
        Format: NNNN_TT_EE_AAA.wav
        where EE is the emotion code (ANG, DIS, FEA, HAP, NEU, SAD)
        """
        try:
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                emotion_code = parts[2].upper()
                return CREMAD_EMOTION_MAP.get(emotion_code)
        except Exception as e:
            logger.warning(f"Failed to parse CREMA-D filename {filename}: {e}")
        return None
    
    @staticmethod
    def process(dataset_path: str, output_dir: str) -> Dict[str, int]:
        """Process CREMA-D dataset."""
        stats = {emotion: 0 for emotion in TARGET_EMOTIONS}
        stats['skipped'] = 0
        
        cremad_path = Path(dataset_path) / 'CREMA-D'
        
        if not cremad_path.exists():
            logger.warning(f"CREMA-D path not found: {cremad_path}")
            return stats
        
        logger.info(f"Processing CREMA-D from {cremad_path}")
        
        # Iterate through all .wav files (typically in flat or nested structure)
        for wav_file in cremad_path.rglob('*.wav'):
            emotion = CremadProcessor.get_emotion(wav_file.name)
            
            if emotion is None:
                stats['skipped'] += 1
                continue
            
            # Create new filename: cremad_<original>.wav
            new_name = f"cremad_{wav_file.name}"
            dest_path = Path(output_dir) / emotion / new_name
            
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(wav_file, dest_path)
                stats[emotion] += 1
            except Exception as e:
                logger.error(f"Failed to copy {wav_file} to {dest_path}: {e}")
        
        return stats


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def combine_datasets(
    data_dir: str,
    output_dir: str = 'combined_dataset',
    skip_existing: bool = False
) -> None:
    """
    Combine all datasets into a single directory structure.
    
    Args:
        data_dir: Path to directory containing RAVDESS, TESS, CREMA-D folders
        output_dir: Output directory path
        skip_existing: If True, skip copying files that already exist
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Validate input directory
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Starting dataset combination...")
    logger.info(f"Input directory: {data_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Target emotions: {TARGET_EMOTIONS}")
    
    # Create output directory structure
    if output_path.exists() and not skip_existing:
        logger.warning(f"Output directory already exists: {output_path}")
        response = input("Continue and overwrite? (y/n): ").lower()
        if response != 'y':
            logger.info("Operation cancelled.")
            return
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize emotion folders
    for emotion in TARGET_EMOTIONS:
        (output_path / emotion).mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    all_stats = {}
    
    logger.info("\n" + "="*70)
    ravdess_stats = RavdessProcessor.process(str(data_path), str(output_path))
    all_stats['RAVDESS'] = ravdess_stats
    logger.info(f"✓ RAVDESS complete: {sum(v for k, v in ravdess_stats.items() if k != 'skipped')} files")
    
    logger.info("\n" + "="*70)
    tess_stats = TessProcessor.process(str(data_path), str(output_path))
    all_stats['TESS'] = tess_stats
    logger.info(f"✓ TESS complete: {sum(v for k, v in tess_stats.items() if k != 'skipped')} files")
    
    logger.info("\n" + "="*70)
    cremad_stats = CremadProcessor.process(str(data_path), str(output_path))
    all_stats['CREMA-D'] = cremad_stats
    logger.info(f"✓ CREMA-D complete: {sum(v for k, v in cremad_stats.items() if k != 'skipped')} files")
    
    # Print summary
    print_summary(all_stats, output_path)


def print_summary(all_stats: Dict, output_path: Path) -> None:
    """Print processing summary."""
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    
    total_by_emotion = {emotion: 0 for emotion in TARGET_EMOTIONS}
    total_skipped = 0
    
    for dataset, stats in all_stats.items():
        logger.info(f"\n{dataset}:")
        for emotion in TARGET_EMOTIONS:
            count = stats[emotion]
            total_by_emotion[emotion] += count
            logger.info(f"  {emotion:12s}: {count:4d}")
        logger.info(f"  {'(skipped)':12s}: {stats['skipped']:4d}")
        total_skipped += stats['skipped']
    
    logger.info("\n" + "-"*70)
    logger.info("COMBINED TOTALS:")
    for emotion in TARGET_EMOTIONS:
        logger.info(f"  {emotion:12s}: {total_by_emotion[emotion]:4d}")
    logger.info(f"  {'(skipped)':12s}: {total_skipped:4d}")
    
    # Verify output directory
    logger.info("\n" + "-"*70)
    logger.info("Output directory structure:")
    for emotion in TARGET_EMOTIONS:
        emotion_dir = output_path / emotion
        if emotion_dir.exists():
            count = len(list(emotion_dir.glob('*.wav')))
            logger.info(f"  {emotion}/: {count} files")
    
    logger.info("\n✓ Dataset combination complete!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Get paths from arguments or use defaults
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'combined_dataset'
    else:
        # Default: assume data directory is at ../data
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / 'data'
        output_dir = script_dir.parent / 'combined_dataset'
    
    combine_datasets(str(data_dir), str(output_dir))
