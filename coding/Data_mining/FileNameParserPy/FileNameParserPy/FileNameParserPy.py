import shutil
import os


#source = 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/Simple_LSTM/RNN-local/RNN-Tutorial/data/raw/librivox/LibriSpeech/temp .txt files/'
source = 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/Simple_LSTM/RNN-local/RNN-Tutorial/data/raw/librivox/LibriSpeech/temp .txt files/'
#dest = 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/Simple_LSTM/RNN-local/RNN-Tutorial/data/raw/librivox/LibriSpeech/train-clean-100-wav/'
dest = 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/Simple_LSTM/RNN-local/RNN-Tutorial/data/raw/librivox/LibriSpeech/kamran/'

temp_folder = 'C:/Users/Mavericks/Documents/GitHub/Mavericks/coding/Simple_LSTM/RNN-local/RNN-Tutorial/data/raw/librivox/LibriSpeech/temp .txt files/copy/'

src_sourcefiles = os.listdir(dest)

for i in range (len(src_sourcefiles)):

    temp = src_sourcefiles[i]
    temp = temp[:1]

    if temp == '0':
        txt_name= '0.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '1':
        txt_name= '1.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '2':
        txt_name= '2.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '3':
        txt_name= '3.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '4':
        txt_name= '4.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '5':
        txt_name= '5.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '6':
        txt_name= '6.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '7':
        txt_name= '7.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '8':
        txt_name= '8.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)

    elif temp == '9':
        txt_name= '9.txt'
        path_temp_with_name = source + txt_name
        shutil.copy(path_temp_with_name,temp_folder)
        new_name = src_sourcefiles[i]
        new_name = new_name[:len(new_name)-4]    # cutting the file type(.wav)
        new_name +='.txt'
        temp_path = temp_folder+txt_name
        shutil.move(temp_path,temp_folder+new_name)     #renaming
        final_move_path = temp_folder + new_name     
        shutil.copy(final_move_path,dest)
        os.remove(final_move_path)


print("Data processing done!")
