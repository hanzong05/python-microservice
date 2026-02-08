#!/usr/bin/env python3
"""
Clean All Scripts - Remove Emojis from All Python Files
Cross-platform script to clean all project files at once
"""

import shutil
from pathlib import Path
import re


def remove_emojis_from_text(text):
    """Remove emojis and replace with text equivalents"""

    # Replace common emojis with text equivalents
    replacements = {
        '✓': '[OK]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '⚠': '[WARNING]',
        'ℹ': '[INFO]',
        '📊': '',
        '📤': '',
        '📥': '',
        '📋': '',
        '📄': '',
        '📦': '',
        '🔍': '',
        '💾': '',
        '🎯': '',
        '⚡': '',
        '🚀': '',
        '🔧': '',
        '⏰': '',
        '📈': '',
        '📉': '',
        '🌐': '',
        '💻': '',
        '🗂': '',
        '📂': '',
        '📁': '',
        '🔐': '',
        '🔑': '',
        '⭐': '',
        '🎨': '',
        '🔬': '',
        '📝': '',
    }

    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)

    # Remove any remaining emoji characters
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\u2600-\u26FF"
        u"\u2700-\u27BF"
        "]+",
        flags=re.UNICODE
    )

    return emoji_pattern.sub('', text)


def clean_file(file_path):
    """Clean a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    cleaned = remove_emojis_from_text(content)

    output_path = file_path.parent / \
        f"{file_path.stem}_no_emojis{file_path.suffix}"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned)

    return output_path


def main():
    print("=" * 70)
    print("CLEANING ALL PYTHON SCRIPTS - REMOVING EMOJIS")
    print("=" * 70)
    print()

    # Files to clean
    files_to_clean = [
        '01_data_cleaning.py',
        '01b_ml_data_preparation.py',
        '02_etl_to_supabase.py',
        '03_feature_engineering.py',
        '04_model_training.py',
        'main.py',
        'run_training_direct.py',
    ]

    current_dir = Path('.')
    cleaned_files = []
    missing_files = []

    print("Step 1: Cleaning files...")
    print()

    for filename in files_to_clean:
        file_path = current_dir / filename

        if file_path.exists():
            try:
                output = clean_file(file_path)
                cleaned_files.append((file_path, output))
                print(f"[OK] Cleaned: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to clean {filename}: {e}")
        else:
            missing_files.append(filename)
            print(f"[WARNING] Not found: {filename}")

    print()
    print(f"Cleaned: {len(cleaned_files)} files")

    if missing_files:
        print(f"Missing: {len(missing_files)} files")

    if not cleaned_files:
        print("\n[ERROR] No files were cleaned!")
        return

    print()
    print("=" * 70)
    print("STEP 2: BACKUP AND REPLACE")
    print("=" * 70)
    print()
    print("This will:")
    print("  1. Backup original files to 'backup_with_emojis/' folder")
    print("  2. Replace originals with cleaned versions")
    print()

    response = input("Continue? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("\nCancelled. Cleaned files saved with '_no_emojis' suffix.")
        return

    # Create backup directory
    backup_dir = current_dir / 'backup_with_emojis'
    backup_dir.mkdir(exist_ok=True)

    print()
    print("Creating backups...")

    for original, cleaned in cleaned_files:
        # Backup original
        backup_path = backup_dir / original.name
        shutil.copy2(original, backup_path)
        print(f"[OK] Backed up: {original.name}")

    print()
    print("Replacing with cleaned versions...")

    for original, cleaned in cleaned_files:
        shutil.copy2(cleaned, original)
        print(f"[OK] Replaced: {original.name}")

    print()
    print("=" * 70)
    print("[SUCCESS] ALL FILES CLEANED AND REPLACED!")
    print("=" * 70)
    print()
    print(f"Originals backed up in: {backup_dir}/")
    print("Cleaned files are now active")
    print()
    print("You can now run your pipeline:")
    print("  python main.py")
    print("  or")
    print("  uvicorn main:app --reload")
    print()


if __name__ == "__main__":
    main()
