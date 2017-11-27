# Designed to work with the OpenSLR LibriSpeech ASR corpus (which is in .flac)
# Requires pydub
# Script that goes in a folder and all subfolders and converts .flac files into .wav files. Can be imported as module or run directly from command line.
# Input format on cmd line: "python FlacToWavPy.py 'source' 'dest'"         
# Source and dest(without apostrphes) are absolute paths to the folder containing the .flac files and the folder to put the converted .wav files respecively 

import os
import sys
from pydub import AudioSegment

class raiseExcept(Exception):
    pass

def flac_check(input_name):
    ''' Function that checks if a file is .flac or not. Returns True or False (bool)
        Parameter: 'input_name' string that has the name of the file(with format). It may work with absolute path and name (maybe) '''    

    _, file_extension = os.path.splitext(input_name)
        
    if file_extension == '.flac':   
        return True
    else:
        return False

def flac_to_wav(input_flac,input_name,dest):
    ''' Function that converts a .flac to a .wav file. Parameter: 'input_flac' - string abs path to a .flac, 
        'input_name' - string of only the name of the .flac (with format), 'dest' string abs path to destination folder (with a '/' in the end!) '''

    audio_file = AudioSegment.from_file(input_flac)
    temp_name = dest
    temp_name += input_name
    temp_name = temp_name[:len(temp_name)-4]
    temp_name += "wav"
    audio_file.export(temp_name)
    print(("File %s conv success!")%(input_name))

def folder_navigator_convert(source, dest):
    ''' Function that goes in a folder and all subfolders and converts .flac files into .wav files.
        Parameters: 'source' - string, abs path to the root folder, 'dest' - string, abs path to dest folder (both must be with a '/' at the end!) '''

    for root, dirs, files in os.walk(source):
        for name in files:
            abs_path = os.path.join(root,name)

            if '\\' in abs_path:
               abs_path = abs_path.replace('\\','/')

            if flac_check(name):
                flac_to_wav(abs_path,name,dest)
            else:
                print(("File %s ignored!")%(name))

if __name__ == "__main__":
    
    try: 
        numb_of_args=len(sys.argv)-1

        if not numb_of_args == 2: 
            raise raiseExcept()
            
    except(raiseExcept):
        print("Number of input parameters wrong! Correct format: 'source' 'dest'")
        sys.exit()

    try: 
        #source_path  = 'C:\\Users\Mavericks\Documents\GitHub\Mavericks\coding\PyTests\FolderNavTest\Testing_folder'             //these two could be used instead of sys.argv[1] and sys.argv[2]
        #dest_path = 'C:\\Users\Mavericks\Documents\GitHub\Mavericks\coding\PyTests\FolderNavTest\Results_folder'                // in order to run the code in an IDE or similar 
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

        folder_navigator_convert(source_path,dest_path)

    except(raiseExcept): 
        print("Wrong parameter format! Correct format: 'source' 'dest'")
    
