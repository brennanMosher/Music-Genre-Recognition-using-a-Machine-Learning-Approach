import Audio_splicing
import Spectogram_Conversion

genre = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for g in genre:
    print("start",g, end='')
    print('   ', end='')

    for i in range(70, 100):
        print('\b', end='')
        print(i, end='')
        if (i<10):
            x = '0'+str(i)
            print('\b', end='')
            print(i, end='')
        else:
            x = str(i)
            print('\b\b', end='')
            print(i, end='')

        Audio_splicing.splice("genres_original/"+g+"/"+g+"_train/"+g+".000"+x+".wav", "Spliced Spectrogram/"+g+"/"+g+"_train/")

        for j in range(10):
            Spectogram_Conversion.spectrogram_conversion("Spliced Spectrogram/"+g+"/"+g+"_train/","Spliced Spectrogram/"+g+"/"+g+"_train/",g+".000"+x+"_splice"+str(j)+".wav",True)
    print()