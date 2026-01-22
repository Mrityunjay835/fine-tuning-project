# from datasets import load_dataset
# import os

# # Load the "imdb" dataset
# ds = load_dataset("databricks/databricks-dolly-15k")

# # Create the folder if it doesn't exist
# output_folder = "data"
# os.makedirs(output_folder, exist_ok=True)

# # To load a specific split (e.g., only the training data)
# # train_data = load_dataset("imdb", split="train")
# print(ds['train'][0]) # Prints the first example of the training data

# # Save the file inside the folder
# ds['train'].to_json(f"{output_folder}/datasets.json")
