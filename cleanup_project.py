#!/usr/bin/env python3
"""
===========================================================
cleanup_project.py

Project Cleanup Script

Removes redundant and test files to streamline the project
structure for the fused topology implementation.

Files to remove:
- Old version backups (v1, v2, v3, v4, v4_1, etc.)
- Duplicate/superseded controllers
- Test result files
- Backup files
===========================================================
"""

import os
import shutil
from pathlib import Path


# Files and directories to remove from root
ROOT_REMOVALS = [
    "jacobian_backup_before_neural_wls.py",
    "run_neural_wls_demo_before_auto_window.py",
    "run_neural_wls_demo_before_demo_fix.py",
    "pmu_simulator_fault_refactored_timing_separated(3).py",
]

# Files to remove from neural_controller/
NEURAL_CONTROLLER_REMOVALS = [
    # Old version test files
    "run_v2_tests.py",
    "run_v3_tests.py",
    "run_v4_tests.py",
    "run_v4_1_tests.py",
    "run_v4_2_tests.py",

    # Backup feature extractors
    "feature_extractor_v1_backup.py",

    # Old version controllers
    "multitask_active_controller.py",
    "multitask_active_controller_v2.py",
    "multitask_active_controller_v3.py",
    "multitask_active_controller_v4.py",
    "multitask_active_controller_v4_1.py",

    # Test result files (v2)
    "test_bad_data_v2.csv",
    "test_clock_drift_v2.csv",
    "test_normal_v2.csv",
    "test_sync_v2.csv",
    "pmu2_bad_data_neural_result.csv",
    "pmu2_clock_drift_neural_result.csv",
    "pmu2_sync_neural_result.csv",
    "normal_neural_result.csv",

    # Test result files (v3)
    "test_bad_data_v3.csv",
    "test_clock_drift_v3.csv",
    "test_normal_v3.csv",
    "test_sync_v3.csv",

    # Test result files (v4)
    "test_bad_data_v4.csv",
    "test_clock_drift_v4.csv",
    "test_normal_v4.csv",
    "test_sync_v4.csv",

    # Test result files (v4.1)
    "test_bad_data_v41.csv",
    "test_clock_drift_v41.csv",
    "test_normal_v41.csv",
    "test_sync_v41.csv",

    # Test result files (v4.2)
    "test_bad_data_v42.csv",
    "test_clock_drift_v42.csv",
    "test_normal_v42.csv",
    "test_sync_v42.csv",

    # Old model versions
    "neural_active_controller.joblib",
    "neural_active_controller.json",
    "neural_active_controller_v2.joblib",
    "neural_active_controller_v2.json",
    "neural_active_controller_v2.dataset.csv",
    "neural_active_controller_v3.joblib",
    "neural_active_controller_v3.json",
    "neural_active_controller_v3.dataset.csv",
    "neural_active_controller_v4.joblib",
    "neural_active_controller_v4.json",
    "neural_active_controller_v4.dataset.csv",
    "neural_active_controller_v41.joblib",
    "neural_active_controller_v41.json",
    "neural_active_controller_v41.dataset.csv",
    "neural_fault_controller.joblib",
    "neural_fault_controller.json",
    "neural_fault_controller.dataset.csv",

    # Expanded test data files
    "expanded_bad_data__pmu1.csv",
    "expanded_bad_data__pmu2.csv",
    "expanded_bad_data__pmu3.csv",
    "expanded_clock_drift__pmu1.csv",
    "expanded_clock_drift__pmu2.csv",
    "expanded_clock_drift__pmu3.csv",
    "expanded_normal.csv",
    "expanded_sync__pmu1.csv",
    "expanded_sync__pmu2.csv",
    "expanded_sync__pmu3.csv",
    "expanded_wls_bad_data__pmu1.csv",
    "expanded_wls_bad_data__pmu2.csv",
    "expanded_wls_bad_data__pmu3.csv",
    "expanded_wls_clock_drift__pmu1.csv",
    "expanded_wls_clock_drift__pmu2.csv",
    "expanded_wls_clock_drift__pmu3.csv",
    "expanded_wls_normal.csv",
    "expanded_wls_sync__pmu1.csv",
    "expanded_wls_sync__pmu2.csv",
    "expanded_wls_sync__pmu3.csv",

    # Test data files (without version)
    "test_bad_data.csv",
    "test_clock_drift.csv",
    "test_normal.csv",
    "test_sync.csv",

    # PMU controller results
    "PMU2_bad_data_r03_controller.csv",
    "PMU2_clock_drift_r03_controller.csv",
    "PMU2_sync_r03_controller.csv",

    # README from refactor
    "README_REFACTOR.txt",
]

# Files to KEEP (active versions)
KEEP_FILES = {
    "neural_controller": [
        "neural_active_controller_v42.joblib",
        "neural_active_controller_v42.json",
        "neural_active_controller_v42.dataset.csv",
        "feature_extractor.py",
        "feature_extractor_v2.py",
        "feature_extractor_v4.py",
        "feature_extractor_v4_1.py",
        "action_controller.py",
        "active_controller.py",
        "multitask_active_controller_v4_2.py",
        "wls_neural.py",
        "train_neural_controller.py",
        "train_multitask_controller.py",
        "train_multitask_controller_v2.py",
        "train_multitask_controller_v3.py",
        "train_multitask_controller_v4.py",
        "train_multitask_controller_v4_1.py",
        "train_multitask_controller_v4_2.py",
    ],
}


def confirm_deletion():
    """Ask user for confirmation before deletion."""
    print("\n" + "=" * 80)
    print(" PROJECT CLEANUP")
    print("=" * 80)

    print("\nThis script will remove:")
    print(f"  - {len(ROOT_REMOVALS)} files from root directory")
    print(f"  - {len(NEURAL_CONTROLLER_REMOVALS)} files from neural_controller/")

    print("\nFiles to remove from root:")
    for f in ROOT_REMOVALS:
        print(f"  - {f}")

    print("\nFiles to remove from neural_controller/:")
    for f in NEURAL_CONTROLLER_REMOVALS:
        print(f"  - {f}")

    response = input("\nProceed with cleanup? (y/n): ").strip().lower()
    return response == 'y'


def cleanup():
    """Perform cleanup operations."""
    base_dir = Path("/home/ankesh/Documents/IIT_Jammu/project_folder/DSSE-PMU-Neural")

    if not confirm_deletion():
        print("\nCleanup cancelled.")
        return

    removed_count = 0
    failed_count = 0

    # Remove from root
    print("\nRemoving from root directory...")
    for filename in ROOT_REMOVALS:
        filepath = base_dir / filename
        try:
            if filepath.exists():
                if filepath.is_file():
                    filepath.unlink()
                    print(f"  ✓ {filename}")
                    removed_count += 1
                elif filepath.is_dir():
                    shutil.rmtree(filepath)
                    print(f"  ✓ {filename}/ (directory)")
                    removed_count += 1
        except Exception as e:
            print(f"  ✗ {filename} - Error: {e}")
            failed_count += 1

    # Remove from neural_controller
    neural_dir = base_dir / "neural_controller"
    print("\nRemoving from neural_controller/...")
    for filename in NEURAL_CONTROLLER_REMOVALS:
        filepath = neural_dir / filename
        try:
            if filepath.exists():
                if filepath.is_file():
                    filepath.unlink()
                    print(f"  ✓ {filename}")
                    removed_count += 1
                elif filepath.is_dir():
                    shutil.rmtree(filepath)
                    print(f"  ✓ {filename}/ (directory)")
                    removed_count += 1
        except Exception as e:
            print(f"  ✗ {filename} - Error: {e}")
            failed_count += 1

    # Summary
    print("\n" + "=" * 80)
    print(" CLEANUP SUMMARY")
    print("=" * 80)
    print(f"Files removed  : {removed_count}")
    print(f"Failed removals: {failed_count}")

    if failed_count == 0:
        print("\n✓ Cleanup completed successfully!")
    else:
        print(f"\n⚠ Cleanup completed with {failed_count} error(s)")


if __name__ == "__main__":
    cleanup()
