import csv
import json

# Specify file paths
csv_file_path = 'DS_Project\\app\data\\World Bank Dataset.csv'
json_file_path = 'DS_Project\\app\data\\output.json'

# Read CSV and convert to JSON
data = []
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(row)

# Write to JSON file
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"CSV has been converted to JSON and saved at {json_file_path}")
