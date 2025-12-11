with open('./original/dataset_seq_test777.tsv', 'r', encoding='utf-8') as infile:
    output_lines = []
    
    for line in infile:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        action = parts[-2]
        obj = parts[-1]
        combined_type = f"{action}"
        
        sub_events = parts[:-2]
        for se in sub_events:
            output_lines.append(f"{se}\t{combined_type}\n")

with open('dataset_seq_test777.txt', 'w', encoding='utf-8') as outfile:
    outfile.writelines(output_lines)