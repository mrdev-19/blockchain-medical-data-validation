import pandas as pd
from ipfshttpclient import connect
from hashlib import sha256
import json
import os
import csv

# Connect to the local IPFS daemon
ipfs = connect("/ip4/127.0.0.1/tcp/5001/http")

# Function to calculate SHA-256 hash of a block
def calculate_hash(data, previous_hash):
    block_string = json.dumps(data, sort_keys=True) + previous_hash
    return sha256(block_string.encode()).hexdigest()

# Function to add data to the IPFS and return CID
def add_to_ipfs(data):
    file_path = "temp.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    result = ipfs.add(file_path)
    os.remove(file_path)
    return result["Hash"]

# Function to read data from CSV and add to the blockchain
def add_to_chain(csv_file):
    df = pd.read_csv(csv_file)

    # Initialize the blockchain with the genesis block
    blockchain = [{"hash": "0" * 64, "data": "Genesis Block"}]

    # Iterate through the CSV data and create blocks
    for i in range(0, len(df), 10):
        records = df.iloc[i:i + 10]
        block_data = records.to_dict(orient="records")
        previous_block = blockchain[-1]
        previous_hash = previous_block["hash"]

        # Calculate hash for the new block
        block_hash = calculate_hash(block_data, previous_hash)

        # Add block data to IPFS and get CID
        cid = add_to_ipfs(block_data)

        # Print debug information
        print(f"Block #{i // 10 + 1}")
        print(f"Previous Hash: {previous_block['hash']}")
        print(f"Calculated Hash: {block_hash}")
        print(f"IPFS CID: {cid}")

        # Create a new block with hash and CID
        new_block = {"hash": block_hash, "data": cid}
        blockchain.append(new_block)
    return blockchain

# Function to check the integrity of the blockchain
def check_integrity(blockchain):
    print("\n\nChecking the Integrity of the blockchain\n\n")
    for i in range(2, len(blockchain)):
        previous_block = blockchain[i - 1]
        current_block = blockchain[i]

        # Print block information for debugging
        print(f"Previous Hash: {previous_block['hash']}")
        print(f"Current Data CID: {current_block['data']}")

        # Retrieve the actual data from IPFS using the CID
        data_cid = current_block['data']
        actual_data_bytes = ipfs.cat(data_cid)
        
        # Convert bytes data to string
        actual_data = actual_data_bytes.decode('utf-8')

        try:
            # Load the actual data as JSON
            actual_data_json = json.loads(actual_data)
        except json.JSONDecodeError:
            # Handle the case where the data is not valid JSON
            return False

        # Calculate hash of the actual data
        current_data_hash = calculate_hash(actual_data_json, previous_block['hash'])

        # Verify if the previous block's hash matches the calculated hash of the actual data
        if current_data_hash != current_block["hash"]:
            return False

    return True

def add_data_to_blockchain_and_csv(df, csv_file_path="output.csv", batch_size=10):
    # Find the last block in the current blockchain
    previous_block = blockchain[-1]
    previous_hash = previous_block["hash"]

    # Split the DataFrame into batches of batch_size
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        chunk = df.iloc[start:end]

        # Calculate hash for the new block
        block_data = chunk.to_dict(orient="records")
        block_hash = calculate_hash(block_data, previous_hash)

        # Add block data to IPFS and get CID
        cid = add_to_ipfs(block_data)

        # Create a new block with hash and CID
        new_block = {"hash": block_hash, "data": cid}
        blockchain.append(new_block)

        # Update CSV file with the new block information
        with open(csv_file_path, mode='a', newline='') as file:
            fieldnames = ["Block Number", "Previous Hash", "Calculated Hash", "IPFS CID"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            block_number = len(blockchain) - 1

            writer.writerow({
                "Block Number": block_number,
                "Previous Hash": previous_block['hash'],
                "Calculated Hash": block_hash,
                "IPFS CID": cid
            })

        print(f"Data added to the blockchain and CSV file: Block #{block_number}")

def csvdata(new_data_df):
    # Add the new data from the DataFrame to the blockchain and update the CSV file
    add_data_to_blockchain_and_csv(new_data_df, blockchain, csv_file_path, batch_size=10)

def store_to_csv(blockchain, csv_file_path="output.csv"):
    with open(csv_file_path, mode='w', newline='') as file:
        fieldnames = ["Block Number", "Previous Hash", "Calculated Hash", "IPFS CID"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(1, len(blockchain)):
            block = blockchain[i]
            previous_block = blockchain[i - 1]

            writer.writerow({
                "Block Number": i,
                "Previous Hash": previous_block['hash'],
                "Calculated Hash": block['hash'],
                "IPFS CID": block['data']
            })

    print(f"Blockchain data stored in {csv_file_path}")

# Example usage
def main():
    csv_file_path = "PatientData.csv"
    blockchain = add_to_chain(csv_file_path)

    # Check integrity of the blockchain
    is_integrity_verified = check_integrity(blockchain)

    if is_integrity_verified:
        print("Blockchain integrity verified.")
        store_to_csv(blockchain)
    else:
        print("Blockchain integrity compromised.")

main()
