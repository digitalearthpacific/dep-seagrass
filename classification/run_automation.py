import os
from pathlib import Path
import papermill as pm

# --- Configuration ---
# Path to your original notebook
notebook_path = "1-generate-postcard-csv.ipynb"

# Folder containing your input GeoJSONs (the "postcards" subfolder)
input_geojson_folder = "training-data/postcards"

# Base folder where output notebooks and CSVs will be saved
output_base_folder = "automated_notebook_runs"
output_csv_subfolder = os.path.join(output_base_folder, "sampled_data_csvs")
output_notebook_subfolder = os.path.join(output_base_folder, "executed_notebooks")

# --- Create Output Directories ---
Path(output_csv_subfolder).mkdir(parents=True, exist_ok=True)
Path(output_notebook_subfolder).mkdir(parents=True, exist_ok=True)
print(f"Output CSVs will be saved to: {output_csv_subfolder}")
print(f"Executed notebooks will be saved to: {output_notebook_subfolder}")

# --- Process Each GeoJSON File ---
print(f"\nStarting automated execution of '{notebook_path}' for GeoJSONs in: {input_geojson_folder}")

processed_count = 0
skipped_count = 0

for filename in os.listdir(input_geojson_folder):
    if filename.endswith(".geojson") and os.path.isfile(os.path.join(input_geojson_folder, filename)):
        current_input_geojson_path = os.path.join(input_geojson_folder, filename)
        
        # Define the output paths for the current run
        # The output CSV name will be based on the GeoJSON filename
        output_csv_name = filename.replace(".geojson", "_sampled_data.csv")
        current_output_csv_path = os.path.join(output_csv_subfolder, output_csv_name)

        # Each executed notebook will also be saved for debugging/record-keeping
        output_notebook_name = filename.replace(".geojson", "_executed.ipynb")
        current_output_notebook_path = os.path.join(output_notebook_subfolder, output_notebook_name)

        print(f"\n--- Processing {filename} ---")
        print(f"  Input GeoJSON: {current_input_geojson_path}")
        print(f"  Output CSV will be: {current_output_csv_path}")
        print(f"  Executed notebook will be: {current_output_notebook_path}")

        try:
            # Execute the notebook using papermill
            pm.execute_notebook(
                input_path=notebook_path,
                output_path=current_output_notebook_path,
                parameters={
                    "input_geojson_path": current_input_geojson_path,
                    "output_csv_path": current_output_csv_path
                }
            )
            print(f"  Successfully executed notebook for {filename}.")
            processed_count += 1

        except Exception as e:
            print(f"  Error executing notebook for {filename}: {e}")
            print(f"  Skipping {filename}.")
            skipped_count += 1
    elif os.path.isdir(os.path.join(input_geojson_folder, filename)):
        print(f"  Skipping directory: {filename}")
    else:
        print(f"  Skipping non-GeoJSON file: {filename}")

print(f"\nAutomation complete. Processed {processed_count} files, skipped {skipped_count} files.")
print(f"Check '{output_csv_subfolder}' for your sampled CSVs and '{output_notebook_subfolder}' for executed notebooks.")
