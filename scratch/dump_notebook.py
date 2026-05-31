import json

def main():
    nb_path = "91093_A03_Project3_Vogel.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Notebook keys: {nb.keys()}")
    cells = nb.get("cells", [])
    print(f"Total cells: {len(cells)}")
    
    code_count = 0
    markdown_count = 0
    
    with open("scratch/notebook_dump.txt", "w", encoding="utf-8") as out:
        for idx, cell in enumerate(cells):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            
            if cell_type == "code":
                code_count += 1
                out.write(f"\n--- CELL {idx} (CODE) ---\n")
                out.write(source)
                out.write("\n")
            elif cell_type == "markdown":
                markdown_count += 1
                out.write(f"\n--- CELL {idx} (MARKDOWN) ---\n")
                out.write(source)
                out.write("\n")
                
    print(f"Dumped {code_count} code cells and {markdown_count} markdown cells to scratch/notebook_dump.txt")

if __name__ == '__main__':
    main()
