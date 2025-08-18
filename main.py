# import os
# import time
# import subprocess
# import sys

# def main_pipeline():
#     """
#     Orchestrates the full ClassID within-session ID assignment pipeline by
#     executing each numbered script in sequence.
#     """
#     print("="*60)
#     print("      ClassID Within-Session ID Assignment Pipeline      ")
#     print("="*60)

#     # --- Configuration: Define all file paths for the pipeline ---
    
#     # Input
#     VIDEO_INPUT_PATH = './input/video.mp4'
    
#     # List of scripts to run in order
#     pipeline_scripts = [
#         '1_oc_sort.py',
#         '2_filter_ephemeral.py',
#         '3_local_reconcil.py'
#     ]

#     # --- Pre-flight Check ---
#     if not os.path.exists(VIDEO_INPUT_PATH):
#         print(f"\n[FATAL ERROR] Input video not found at '{VIDEO_INPUT_PATH}'")
#         print("Please create an 'input' folder and place your 'video.mp4' file inside.")
#         return
        
#     for script in pipeline_scripts:
#         if not os.path.exists(script):
#             print(f"\n[FATAL ERROR] Pipeline script not found: '{script}'")
#             print("Please ensure all scripts (1_oc_sort.py, 2_filter_ephemeral.py, 3_local_reconcil.py) are in the same directory.")
#             return

#     total_start_time = time.time()
    
#     # --- Execute each script in the pipeline ---
#     for script in pipeline_scripts:
#         print(f"\n>>> Executing Step: {script}...")
#         step_start_time = time.time()
        
#         try:
#             # We use `sys.executable` to ensure we're using the same Python interpreter
#             # that is running main.py. This avoids issues with virtual environments.
#             result = subprocess.run(
#                 [sys.executable, script], 
#                 check=True, 
#                 capture_output=True, 
#                 text=True
#             )
#             # Print the output from the script
#             print(result.stdout)
            
#         except subprocess.CalledProcessError as e:
#             print(f"\n[FATAL ERROR] An error occurred while running '{script}'.")
#             print("--- Script Output (stdout) ---")
#             print(e.stdout)
#             print("--- Script Error (stderr) ---")
#             print(e.stderr)
#             print("="*60)
#             print("Pipeline halted.")
#             return

#         step_end_time = time.time()
#         print(f">>> Step '{script}' complete. Duration: {step_end_time - step_start_time:.2f} seconds.")

#     total_end_time = time.time()
#     print("\n" + "="*60)
#     print("Pipeline finished successfully!")
#     print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")
#     print("Final results are in the 'output/' directory.")
#     print("="*60)

# if __name__ == '__main__':
#     main_pipeline()


import os
import time
import subprocess
import sys

def main_pipeline():
    """
    Orchestrates the full ClassID within-session ID assignment pipeline by
    executing each numbered script in sequence.
    """
    print("="*60)
    print("      ClassID Within-Session ID Assignment Pipeline      ")
    print("="*60)

    pipeline_scripts = [
        '1_oc_sort.py',
        '2_filter_ephemeral.py',
        '3_local_reconcil.py',
        '4_feature_extraction.py',
        '5_global_reconcil.py'
    ]

    if not os.path.exists('./input/video.mp4'):
        print("\n[FATAL ERROR] Input video not found at './input/video.mp4'")
        sys.exit(1)

    total_start_time = time.time()
    
    for script in pipeline_scripts:
        print(f"\n>>> Executing Step: {script}...")
        step_start_time = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, script], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("--- Stderr ---")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"\n[FATAL ERROR] An error occurred while running '{script}'.")
            print("--- Script Output (stdout) ---")
            print(e.stdout)
            print("--- Script Error (stderr) ---")
            print(e.stderr)
            print("="*60)
            print("Pipeline halted.")
            sys.exit(1)

        step_end_time = time.time()
        print(f">>> Step '{script}' complete. Duration: {step_end_time - step_start_time:.2f} seconds.")

    total_end_time = time.time()
    print("\n" + "="*60)
    print("Pipeline finished successfully!")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")
    print("Final results are in the 'output/' directory.")
    print("="*60)

if __name__ == '__main__':
    main_pipeline()