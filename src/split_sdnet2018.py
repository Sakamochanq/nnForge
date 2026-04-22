import os
import shutil
import re
from pathlib import Path


def extract_numeric_id(filename):
    """
    Extract numeric ID from filename for proper sorting.
    Supports patterns like:
    - 7069-101.jpg -> 101
    - 7069-104.jpg -> 104
    - 7136-202.jpg -> 202
    
    Returns (numeric_id, original_filename)
    """
    base = Path(filename).stem
    
    # Pattern: numbers-numbers or just numbers
    match = re.search(r'-(\d+)$', base)
    if match:
        numeric_id = int(match.group(1))
        return numeric_id, filename
    
    # Fallback: try to extract any number sequence
    match = re.search(r'(\d+)', base)
    if match:
        numeric_id = int(match.group(1))
        return numeric_id, filename
    
    return float('inf'), filename  # Put files with no numbers at the end


def get_files_by_numeric_order(dataset_dir):
    """
    Get files sorted by numeric ID.
    Returns list of (numeric_id, filename) tuples sorted by ID.
    """
    files = os.listdir(dataset_dir)
    file_list = []
    
    for file in files:
        numeric_id, filename = extract_numeric_id(file)
        file_list.append((numeric_id, filename))
    
    # Sort by numeric ID
    file_list.sort(key=lambda x: x[0])
    return file_list


def split_dataset_class(class_dir, output_base_dir, class_name, train_ratio=0.70, val_ratio=0.15):
    """
    Split a single class directory into train/val/test.
    
    Args:
        class_dir: Input directory for one class (e.g., W/CW or W/UW)
        output_base_dir: Base output directory
        class_name: Class name (CrackWall or UncrackWall)
        train_ratio: Ratio for training set (default: 0.70)
        val_ratio: Ratio for validation set (default: 0.15)
    """
    
    if not os.path.isdir(class_dir):
        print(f"Error: Class directory '{class_dir}' does not exist.")
        return 0, 0, 0
    
    # Get files sorted by numeric order
    file_list = get_files_by_numeric_order(class_dir)
    total_count = len(file_list)
    
    if total_count == 0:
        print(f"  No files found in {class_name}")
        return 0, 0, 0
    
    # Create output directories
    train_dir = os.path.join(output_base_dir, 'train', class_name)
    val_dir = os.path.join(output_base_dir, 'val', class_name)
    test_dir = os.path.join(output_base_dir, 'test', class_name)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Calculate split indices
    train_split = int(total_count * train_ratio)
    val_split = int(total_count * (train_ratio + val_ratio))
    
    print(f"\n  Class: {class_name}")
    print(f"  Total files: {total_count}")
    print(f"  Sequential split indices: Train[0~{train_split-1}] Val[{train_split}~{val_split-1}] Test[{val_split}~{total_count-1}]")
    
    # Split and copy files
    train_count = 0
    val_count = 0
    test_count = 0
    
    for idx, (numeric_id, filename) in enumerate(file_list):
        src = os.path.join(class_dir, filename)
        
        # Determine split
        if idx < train_split:
            dst = os.path.join(train_dir, filename)
            train_count += 1
        elif idx < val_split:
            dst = os.path.join(val_dir, filename)
            val_count += 1
        else:
            dst = os.path.join(test_dir, filename)
            test_count += 1
        
        shutil.copy2(src, dst)
    
    print(f"  Result: Train({train_count}) Val({val_count}) Test({test_count})")
    
    return train_count, val_count, test_count


def split_dataset(dataset_dir, output_dir, train_ratio=0.70, val_ratio=0.15):
    """
    Split SDNET2018 dataset into train/val/test for CrackWall and UncrackWall.
    Directory structure: dataset_dir/W/CW/ and dataset_dir/W/UW/
    Output structure: output_dir/train/CrackWall/, output_dir/train/UncrackWall/, etc.
    
    Args:
        dataset_dir: Input directory containing W/ subdirectory
        output_dir: Output directory for train/val/test folders
        train_ratio: Ratio for training set (default: 0.70)
        val_ratio: Ratio for validation set (default: 0.15)
                   Test ratio = 1 - train_ratio - val_ratio
    """
    
    # Validate input
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return
    
    # Create output base directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("SDNET2018 Dataset Split (CrackWall & UncrackWall)")
    print("=" * 70)
    print(f"Input directory:  {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratio - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {(1-train_ratio-val_ratio)*100:.0f}%")
    print()
    
    # Define class directories
    cw_dir = os.path.join(dataset_dir, 'W', 'CW')
    uw_dir = os.path.join(dataset_dir, 'W', 'UW')
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # Process CrackWall
    if os.path.isdir(cw_dir):
        train_c, val_c, test_c = split_dataset_class(
            cw_dir, output_dir, 'CrackWall', train_ratio, val_ratio
        )
        total_train += train_c
        total_val += val_c
        total_test += test_c
    else:
        print(f"Warning: CrackWall directory not found at {cw_dir}")
    
    # Process UncrackWall
    if os.path.isdir(uw_dir):
        train_u, val_u, test_u = split_dataset_class(
            uw_dir, output_dir, 'UncrackWall', train_ratio, val_ratio
        )
        total_train += train_u
        total_val += val_u
        total_test += test_u
    else:
        print(f"Warning: UncrackWall directory not found at {uw_dir}")
    
    # Print summary
    print()
    print("=" * 70)
    print("Split Complete!")
    print("=" * 70)
    print(f"Total files - Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── CrackWall/")
    print(f"  │   └── UncrackWall/")
    print(f"  ├── val/")
    print(f"  │   ├── CrackWall/")
    print(f"  │   └── UncrackWall/")
    print(f"  └── test/")
    print(f"      ├── CrackWall/")
    print(f"      └── UncrackWall/")
    print()


if __name__ == "__main__":
    # Get input from user
    dataset_dir = input("Enter dataset root directory path (e.g., ./dataset/images): ").strip()
    output_dir = input("Enter output directory path: ").strip()
    
    # Validate paths
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"\nInput directory:  {dataset_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Run split
    split_dataset(dataset_dir, output_dir, train_ratio=0.70, val_ratio=0.15)
import os
import shutil
import re
from pathlib import Path
from collections import defaultdict


def extract_numeric_id(filename):
    """
    Extract numeric ID from filename for proper sorting.
    Supports patterns like:
    - 7069-101.jpg -> 101
    - 7069-104.jpg -> 104
    - 7136-202.jpg -> 202
    
    Returns (numeric_id, original_filename)
    """
    base = Path(filename).stem
    
    # Pattern: numbers-numbers or just numbers
    match = re.search(r'-(\d+)$', base)
    if match:
        numeric_id = int(match.group(1))
        return numeric_id, filename
    
    # Fallback: try to extract any number sequence
    match = re.search(r'(\d+)', base)
    if match:
        numeric_id = int(match.group(1))
        return numeric_id, filename
    
    return float('inf'), filename  # Put files with no numbers at the end


def get_files_by_numeric_order(dataset_dir):
    """
    Get files sorted by numeric ID.
    Returns list of (numeric_id, filename) tuples sorted by ID.
    """
    files = os.listdir(dataset_dir)
    file_list = []
    
    for file in files:
        numeric_id, filename = extract_numeric_id(file)
        file_list.append((numeric_id, filename))
    
    # Sort by numeric ID
    file_list.sort(key=lambda x: x[0])
    return file_list



def split_dataset_class(class_dir, output_base_dir, class_name, train_ratio=0.70, val_ratio=0.15):
    """
    Split a single class directory into train/val/test.
    
    Args:
        class_dir: Input directory for one class (e.g., W/CW or W/UW)
        output_base_dir: Base output directory
        class_name: Class name (CrackWall or UncrackWall)
        train_ratio: Ratio for training set (default: 0.70)
        val_ratio: Ratio for validation set (default: 0.15)
    """
    
    if not os.path.isdir(class_dir):
        print(f"Error: Class directory '{class_dir}' does not exist.")
        return 0, 0, 0
    
    # Get files sorted by numeric order
    file_list = get_files_by_numeric_order(class_dir)
    total_count = len(file_list)
    
    if total_count == 0:
        print(f"  No files found in {class_name}")
        return 0, 0, 0
    
    # Create output directories
    train_dir = os.path.join(output_base_dir, 'train', class_name)
    val_dir = os.path.join(output_base_dir, 'val', class_name)
    test_dir = os.path.join(output_base_dir, 'test', class_name)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Calculate split indices
    train_split = int(total_count * train_ratio)
    val_split = int(total_count * (train_ratio + val_ratio))
    
    print(f"\n  Class: {class_name}")
    print(f"  Total files: {total_count}")
    print(f"  Sequential split indices: Train[0~{train_split-1}] Val[{train_split}~{val_split-1}] Test[{val_split}~{total_count-1}]")
    
    # Split and copy files
    train_count = 0
    val_count = 0
    test_count = 0
    
    for idx, (numeric_id, filename) in enumerate(file_list):
        src = os.path.join(class_dir, filename)
        
        # Determine split
        if idx < train_split:
            dst = os.path.join(train_dir, filename)
            train_count += 1
        elif idx < val_split:
            dst = os.path.join(val_dir, filename)
            val_count += 1
        else:
            dst = os.path.join(test_dir, filename)
            test_count += 1
        
        shutil.copy2(src, dst)
    
    print(f"  Result: Train({train_count}) Val({val_count}) Test({test_count})")
    
    return train_count, val_count, test_count


def split_dataset(dataset_dir, output_dir, train_ratio=0.70, val_ratio=0.15):
    """
    Split SDNET2018 dataset into train/val/test for CrackWall and UncrackWall.
    Directory structure: dataset_dir/W/CW/ and dataset_dir/W/UW/
    Output structure: output_dir/train/CrackWall/, output_dir/train/UncrackWall/, etc.
    
    Args:
        dataset_dir: Input directory containing W/ subdirectory
        output_dir: Output directory for train/val/test folders
        train_ratio: Ratio for training set (default: 0.70)
        val_ratio: Ratio for validation set (default: 0.15)
                   Test ratio = 1 - train_ratio - val_ratio
    """
    
    # Validate input
    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.")
        return
    
    # Create output base directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Print header
    print("=" * 70)
    print("SDNET2018 Dataset Split (CrackWall & UncrackWall)")
    print("=" * 70)
    print(f"Input directory:  {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratio - Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {(1-train_ratio-val_ratio)*100:.0f}%")
    print()
    
    # Define class directories
    cw_dir = os.path.join(dataset_dir, 'W', 'CW')
    uw_dir = os.path.join(dataset_dir, 'W', 'UW')
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    # Process CrackWall
    if os.path.isdir(cw_dir):
        train_c, val_c, test_c = split_dataset_class(
            cw_dir, output_dir, 'CrackWall', train_ratio, val_ratio
        )
        total_train += train_c
        total_val += val_c
        total_test += test_c
    else:
        print(f"Warning: CrackWall directory not found at {cw_dir}")
    
    # Process UncrackWall
    if os.path.isdir(uw_dir):
        train_u, val_u, test_u = split_dataset_class(
            uw_dir, output_dir, 'UncrackWall', train_ratio, val_ratio
        )
        total_train += train_u
        total_val += val_u
        total_test += test_u
    else:
        print(f"Warning: UncrackWall directory not found at {uw_dir}")
    
    # Print summary
    print()
    print("=" * 70)
    print("Split Complete!")
    print("=" * 70)
    print(f"Total files - Train: {total_train}, Val: {total_val}, Test: {total_test}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── CrackWall/")
    print(f"  │   └── UncrackWall/")
    print(f"  ├── val/")
    print(f"  │   ├── CrackWall/")
    print(f"  │   └── UncrackWall/")
    print(f"  └── test/")
    print(f"      ├── CrackWall/")
    print(f"      └── UncrackWall/")
    print()


if __name__ == "__main__":
    # Get input from user
    dataset_dir = input("Enter dataset root directory path (e.g., ./dataset/images): ").strip()
    output_dir = input("Enter output directory path: ").strip()
    
    # Validate paths
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    
    print(f"\nInput directory:  {dataset_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Run split
    split_dataset(dataset_dir, output_dir, train_ratio=0.70, val_ratio=0.15)
