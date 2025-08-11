import os
import gdown
import zipfile

def download_and_unzip_gdrive(file_id, dest_folder="data"):
    """
    Downloads a file from Google Drive, saves it, and unzips it.

    Args:
        file_id (str): The Google Drive file ID.
        dest_folder (str): The folder to save and extract the file to.
    """
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)

    output_filename = os.path.join(dest_folder, "NTU-Fi_dataset.zip")

    # --- Download the file using gdown ---
    print(f"Downloading dataset from Google Drive (ID: {file_id})...")
    gdown.download(id=file_id, output=output_filename, quiet=False)

    if not os.path.exists(output_filename):
        print("gdown failed to download the file.")
        return

    # --- Unzip the file ---
    print(f"Unzipping {output_filename}...")
    success = False
    try:
        with zipfile.ZipFile(output_filename, 'r') as zip_ref:
            # Get a list of all archived file paths
            file_list = zip_ref.namelist()
            print(f"Extracting {len(file_list)} files to {dest_folder}...")
            zip_ref.extractall(dest_folder)

        print(f"Successfully unzipped to {dest_folder}")
        success = True
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file is not a valid zip file.")
        print(f"The file will be kept for inspection at: {output_filename}")
        return
    except Exception as e:
        print(f"An error occurred during unzipping: {e}")
        return
    finally:
        # --- Clean up the zip file only on success ---
        if success:
            print(f"Cleaning up {output_filename}...")
            os.remove(output_filename)
            print("Cleanup complete.")

if __name__ == "__main__":
    # The ID from the user-provided Google Drive link
    # Per default, only download the limited dataset because of limited storage space
    GDRIVE_FILE_ID = "1ZnDpn9CxoT0hrLSeGV1hcif2DGjg6Xlk"
    download_and_unzip_gdrive(GDRIVE_FILE_ID)
