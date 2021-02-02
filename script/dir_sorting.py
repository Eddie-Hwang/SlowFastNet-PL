import argparse
import os
import glob

import sh

parser = argparse.ArgumentParser()
parser.add_argument('--list_path', type=str, default='./ucfTrainTestlist')
parser.add_argument('--ufc', type=str, default='./dataset')
args = parser.parse_args()

def get_file_list(_list):
    _file_list = []
    for _l in _list:
        with open(_l) as f:
            while True:
                line = f.readline()
                if not line: break
                f_name = os.path.split(line)[-1].split(' ')[0]
                _file_list.append(f_name)
    return _file_list

def mv_files(_file_list, mode):
    for _file in _file_list:
        label = _file.split('_')[1]
        save_path = os.path.join(args.ufc, mode, label)
        if not(os.path.exists(save_path)):
            os.mkdir(save_path)
        origin_path = os.path.join(args.ufc, _file).strip()
        if os.path.exists(origin_path):
            sh.mv(origin_path, save_path)
            print(f'{origin_path} -> {save_path}/{_file}')

def sort_from_raw_data():
    
    train_lists = glob.glob(os.path.join(args.list_path, 'trainlist01.txt'))
    val_lists = glob.glob(os.path.join(args.list_path, 'testlist01.txt'))
    
    tr_file_list = get_file_list(train_lists)
    val_file_list = get_file_list(val_lists)
    
    # mv_files(tr_file_list, 'train')
    mv_files(val_file_list, 'validation')

    print('done.')
    

def main():
    # _, _ = map(mv_files_again, ['train', 'validation'])
    sort_from_raw_data()
    
        
        

if __name__=='__main__':
    main()
