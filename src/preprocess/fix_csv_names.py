import os
directory = 'REAL_RECOVERED_DATA'
for file_name in os.listdir(directory):
    fixed_file_name = file_name[:-4]
    src = os.path.join(directory, file_name)
    tgt = os.path.join(directory, fixed_file_name)
    os.rename(src, tgt)