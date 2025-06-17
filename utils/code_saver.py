import os

# Define the filename for the Python script


def save_code(filename, code):

    # filename = "model_training_gpt.py"

    # Extract the directory path from the full filename
    directory = os.path.dirname(filename)

    # If a directory path exists, ensure it is created
    if directory:
        # Create all necessary intermediate directories.
        # exist_ok=True prevents an error if the directory already exists.
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory '{directory}' exists.")

    # Open the file in write mode ('w').
    # This mode creates the file if it doesn't exist,
    # or truncates (clears) it if it does.

    try:
        # Save the code to a file
        with open(filename, "w") as f:
            f.write(code)
        print(f"Code successfully saved to '{filename}'")

    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")