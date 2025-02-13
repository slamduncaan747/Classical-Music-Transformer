import csv


def process_csv(filename, num_lines=4):
    result = []
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for i, row in enumerate(csv_reader):
            if i >= num_lines:
                break
            result.extend(row)
    return ",".join(result)


# Example usage
if __name__ == "__main__":
    filename = "./train/chorale_000.csv"  # Replace with your CSV file path
    num_lines = 20  # Change this to get more or fewer lines
    output = process_csv(filename, num_lines)
    print(output)
