def read_data() -> list:
    with open("./data/filtered_data.txt", "r") as f:
        lines = f.readlines()
    return lines

def main():
    for l in read_data():
        print(l, end="")
    
if __name__ == "__main__":
    main()