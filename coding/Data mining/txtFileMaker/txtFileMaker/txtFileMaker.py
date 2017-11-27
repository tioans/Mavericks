# Designed to work with the OpenSLR LibriSpeech ASR corpus (which is in .flac)
# Script that goes in a folder and all subfolders and parses one .txt file containg paragraphs to multiple .txt files corresponding to their .wav files
# The result would be: for every .wav with the standard OpenSLR name, one .txt file with the same name as the .wav (audio corresponds to the paragraph)
# Input format on cmd line: "python txtFileMaker.py 'source' 'dest'"         
# Source and dest(without apostrphes) are absolute paths to the folder containing the unsplit .txt files and the folder where the converted .flac to .wav files are respecively 

import os
import sys

class raiseExcept(Exception):
    pass

def txt_check(input_name):
    ''' Function that checks if a file is .txt or not. Returns True or False (bool)
        Parameter: 'input_name' string, that has the name of the file(with format). It may work with absolute path and name (maybe) '''    

    _, file_extension = os.path.splitext(input_name)
        
    if file_extension == '.txt':   
        return True
    else:
        return False

def txt_maker(input_file,input_name,dest):
    ''' Function that splits a .txt file into multiple .txt files so that the .wav files and .txt files are 1 to 1 pairs. 
        Parameter: 'input_file' - string abs path to a .txt, 
        'input_name' - string of only the name of the .txt (with format), 
        'dest' string abs path to destination folder (with a '/' in the end!) '''    

    new_file_counter = 0
    temp_len = len(input_name)
    temp_name = input_name[:temp_len-10]
    
    fo = open (input_file,"r")
    
    str_temp = fo.read()

    char_to_look_for = " "
    ctrl_seq_pos = str_temp.find(temp_name)
    first_space = str_temp.find(char_to_look_for)
        
    while ctrl_seq_pos != -1:
        
        temp_start_content = first_space+1
        new_file_name = dest
        new_file_name += str_temp[ctrl_seq_pos:first_space] + ".txt"     
        
        ctrl_seq_pos = str_temp.find(temp_name,ctrl_seq_pos+20)
                      
        new_fo = open (new_file_name,"w")
        new_file_counter+=1

        if ctrl_seq_pos != -1:        
            new_content = str_temp[temp_start_content:ctrl_seq_pos-1]
            first_space = str_temp.find(char_to_look_for,ctrl_seq_pos)
        else:
            new_content = str_temp[temp_start_content:len(str_temp)]            # could be problematic, maybe
        
        new_fo.write(new_content)

        new_fo.close()

    print(("File %s parsed successfully into %i .txt files!") % (input_name, new_file_counter))

    fo.close()

def add_numb_of_files(input):
    return input+1

# 0 - wav, 1 - txt
def numb_of_files(input,wav_or_txt,count):
    ''' Function that counts number of txt files '''
    if wav_or_txt == 0:
        if txt_check(input):
            count = add_numb_of_files(count)
        
    elif wav_or_txt == 1:
        if not txt_check(input):
            count = add_numb_of_files(count)
    
    return count 

def folder_navigator_txt(source, dest):
    ''' Function that goes in a folder and all subfolders and splits .txt files into multiple .txt files so that the .wav files and .txt files are 1 to 1 pairs.
        Parameters: 'source' - string, abs path to the root folder, 'dest' - string, abs path to dest folder (both must be with a '/' at the end!) '''

    for root, dirs, files in os.walk(source):
        for name in files:
            abs_path = os.path.join(root,name)

            if '\\' in abs_path:
               abs_path = abs_path.replace('\\','/')

            if txt_check(name):
                txt_maker(abs_path,name,dest)

if __name__ == "__main__":
    
    try: 
        numb_of_args=len(sys.argv)-1

        if not numb_of_args == 2: 
            raise raiseExcept()
            
    except(raiseExcept):
        print("Number of input parameters wrong! Correct format: 'source' 'dest'")
        sys.exit()

    try: 
        #source_path  = 'C:\\Users\Mavericks\Documents\GitHub\Mavericks\coding\PyTests\FolderNavTest\Testing_folder'           //these two could be used instead of sys.argv[1] and sys.argv[2]
        #dest_path = 'C:\\Users\Mavericks\Documents\GitHub\Mavericks\coding\PyTests\FolderNavTest\Results_folder'              // in order to run the code in an IDE or similar 

        source_path = sys.argv[1]
        dest_path = sys.argv[2]
        
        if '\\' in source_path:
            source_path = source_path.replace('\\','/')

        if '\\' in dest_path:
            dest_path = dest_path.replace('\\','/')

        if not os.path.isdir(source_path):
            raise raiseExcept()

        if not os.path.isdir(dest_path):
            raise raiseExcept()
        
        if source_path[len(source_path)-1:] is not '/':
            source_path+='/'

        if dest_path[len(dest_path)-1:] is not '/':
            dest_path+='/'

        folder_navigator_txt(source_path,dest_path)

    except(raiseExcept): 
        print("Wrong parameter format! Correct format: 'source' 'dest'")


################################# LEGACY code #########################################

## parameters for find_missing_pairs
## wav_temp =''

## VERY inefficient
#def find_missing_pairs(name, files, missing_pairs_count):
#    i = 0
#    chk = 0

#    if txt_check(name):
#        txt_temp= name[:len(name)-3]
        
#        for i in range(len(files)):
#            if not txt_check(files[i]):
#                tmp = files[i]
#                wav_temp = tmp[:len(tmp)-3]
                
#                if txt_temp == wav_temp:
#                    chk = 1
#                    break
                
#    if chk == 0:
#        missing_pairs_count = add_numb_of_files(missing_pairs_count)
#    else:
#        chk = 0
#    i+=1
