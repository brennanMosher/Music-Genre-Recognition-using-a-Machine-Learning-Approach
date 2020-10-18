from pydub import AudioSegment

# Convert audio file to 1 channel representation
def Stereo_Mono_Conversion(audio_path, audio_file):
    output_path = audio_path + 'Mono\mono_' + audio_file

    print(audio_path + audio_file)
    sound = AudioSegment.from_wav(audio_path + audio_file)
    sound = sound.set_channels(1)
    sound.export(output_path, format='wav')

    return output_path