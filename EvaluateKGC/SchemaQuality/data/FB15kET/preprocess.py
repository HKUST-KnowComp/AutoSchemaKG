def load_entity2text(filename):
    entity_dict = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            entity_id, text = line.strip().split('\t')
            entity_dict[entity_id] = text
    return entity_dict

def process_entity_type(input_file, output_file, entity_dict):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            entity_id, entity_type = line.strip().split('\t')
            entity_text = entity_dict.get(entity_id, entity_id)
            fout.write(f'{entity_text}\t{entity_type}\n')

def process_test(input_file, output_file, entity_dict):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            col1, col2, col3 = line.strip().split('\t')
            text1 = entity_dict.get(col1, col1)
            text3 = entity_dict.get(col3, col3)
            fout.write(f'{text1}\t{col2}\t{text3}\n')

def main():
    # 文件路径
    entity2text_file = './original/entity2text.txt'
    entity_type_test_input = './original/Entity_Type_test.txt'
    entity_type_train_input = './original/Entity_Type_train.txt'
    entity_type_valid_input = './original/Entity_Type_valid.txt'
    test_input = './original/test.txt'
    train_input = './original/train.txt'
    valid_input = './original/valid.txt'

    entity_type_test_output = './Entity_Type_test.txt'
    entity_type_train_output = './Entity_Type_train.txt'
    entity_type_valid_output = './Entity_Type_valid.txt'
    test_output = './test.txt'
    train_output = './train.txt'
    valid_output = './valid.txt'

    entity_dict = load_entity2text(entity2text_file)

    process_entity_type(entity_type_test_input, entity_type_test_output, entity_dict)
    process_entity_type(entity_type_train_input, entity_type_train_output, entity_dict)
    process_entity_type(entity_type_valid_input, entity_type_valid_output, entity_dict)
    process_test(test_input, test_output, entity_dict)
    process_test(train_input, train_output, entity_dict)
    process_test(valid_input, valid_output, entity_dict)

if __name__ == "__main__":
    main()