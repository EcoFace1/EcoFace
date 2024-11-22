import os
import torch
from transformers import Wav2Vec2Processor, HubertModel
import soundfile as sf
import librosa
import numpy as np
import importlib
import tqdm
import sys
import torch
import argparse
from utils.commons.tensor_utils import move_to_cuda
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
from collections import OrderedDict
from modules.EMG.vae import VAEModel
from modules.EDE.model import EmoEmb
import cv2
import time
import glob
from torchvision import transforms
from PIL import Image

def get_pic_data(flamepic):
    pic=sorted(glob.glob(os.path.join(flamepic,'*.png')),key=lambda x:int(x.split('.png')[0].split('/')[-1]))
    pic_data=np.zeros((len(pic), 1, 128, 128))
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49863],std=[0.19283]),
    ]) 
    for i in range(len(pic)):
        img = Image.open(pic[i]).convert('L') #gray, RGB
        img = data_transform(img) #(1,128,128)
        pic_data[i]=img.numpy()
    
    pic_data=pic_data.transpose((1, 0, 2, 3)) #(time,1,128,128) -> (1,time,128,128)
    #return pic
    return pic_data

def audio_driven(arg):
    if not os.path.exists(arg.result_path):
        os.makedirs(arg.result_path)
    #subj_list
    person_dic = np.load(arg.person_dict_path)
    person_dict = {key: value for key, value in person_dic.items()}
    #shape_list
    shape_dic = np.load(arg.shape_dict_path)
    shape_dict = {key: value for key, value in shape_dic.items()}
    #ede_hubert
    print("------loading EDE model-------")
    feature_net = EmoEmb()
    load_ckpt(feature_net, arg.ede_model_path)
    feature_net = feature_net.to(torch.device(arg.device))
    feature_net.eval()
    #hubert
    print("------loading hubert model-------")
    processor = Wav2Vec2Processor.from_pretrained(arg.hubert_path)
    hubert_model = HubertModel.from_pretrained(arg.hubert_path)
    hubert_model =hubert_model.to(torch.device(arg.device))
    hubert_model.eval()
    #emg model
    print("------loading fcg model-------")
    model = VAEModel(in_out_dim=53)
    ckp=torch.load(arg.emg_model_path)['state_dict']['model']
    model.load_state_dict(ckp)
    model = model.to(torch.device(arg.device))
    model.eval()
    #flame model
    from flame_model.flame import FlameHead
    flame=FlameHead(50,100).to(torch.device(arg.device))
    flame.eval()

    wav_path = arg.wav_path
    au_emo_path = arg.audio_emo_path
    
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    input_values_speech = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_speech = input_values_speech.to(torch.device(arg.device))

    emo_array, sampling_rate = librosa.load(os.path.join(au_emo_path), sr=16000)
    input_values_emo = processor(emo_array, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_emo = input_values_emo.to(torch.device(arg.device))

    person_id = torch.tensor(person_dict[arg.condition]).unsqueeze(0).to(torch.device(arg.device))

    #predict
    hubert_feature = hubert_model(input_values_speech).last_hidden_state
    emo_feature = feature_net.net_aud.forward(input_values_emo).last_hidden_state

    if emo_feature.shape[1] < hubert_feature.shape[1]:
        while(emo_feature.shape[1]<hubert_feature.shape[1]):
            temp = emo_feature
            emo_feature = torch.cat([temp,temp], dim=1)
    emo_feature = emo_feature[:, :hubert_feature.shape[1], :]

    pr_audio = model.predict(hubert_feature,emo_feature,person_id) #(1,time,53)

    bt=pr_audio.shape[0]
    tim=pr_audio.shape[1]
    exp = pr_audio[:,:,:-3] #.to(torch.device(arg.device))
    jaw = pr_audio[:,:,-3:] #.to(torch.device(arg.device))
    shape = torch.from_numpy(shape_dict[arg.condition]).unsqueeze(0)[:,None,:].repeat(1, tim, 1).to(torch.device(arg.device))
    rotation = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    neck_pose = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    eyes_pose = torch.zeros(bt,tim, 6).to(torch.device(arg.device))
    verts_audio,ldmark= flame(shape,exp,rotation,neck_pose,jaw,eyes_pose)
    
    pred_audio = verts_audio[0]

    np.save(os.path.join(arg.resulllt_path, 'audio_ref.npy'), pred_audio.detach().cpu().numpy())



def video_driven(arg):
    if not os.path.exists(arg.result_path):
        os.makedirs(arg.result_path)
    # subj_list
    person_dic = np.load(arg.person_dict_path)
    person_dict = {key: value for key, value in person_dic.items()}
    # shape_list
    shape_dic = np.load(arg.shape_dict_path)
    shape_dict = {key: value for key, value in shape_dic.items()}
    # ede_hubert
    print("------loading EDE model-------")
    feature_net = EmoEmb()
    load_ckpt(feature_net, arg.ede_model_path)
    feature_net = feature_net.to(torch.device(arg.device))
    feature_net.eval()
    # hubert
    print("------loading hubert model-------")
    processor = Wav2Vec2Processor.from_pretrained(arg.hubert_path)
    hubert_model = HubertModel.from_pretrained(arg.hubert_path)
    hubert_model = hubert_model.to(torch.device(arg.device))
    hubert_model.eval()
    # emg model
    print("------loading fcg model-------")
    model = VAEModel(in_out_dim=53)
    ckp = torch.load(arg.emg_model_path)['state_dict']['model']
    model.load_state_dict(ckp)  # path of the vggish pretrained network
    model = model.to(torch.device(arg.device))
    model.eval()
    # flame model
    from flame_model.flame import FlameHead
    flame = FlameHead(50, 100).to(torch.device(arg.device))
    flame.eval()

    wav_path = arg.wav_path
    vi_emo_path = arg.video_emo_path

    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    input_values_speech = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_speech = input_values_speech.to(torch.device(arg.device))

    emo_pic = torch.from_numpy(get_pic_data(vi_emo_path)).unsqueeze(0).float().to(torch.device(arg.device))

    person_id = torch.tensor(person_dict[arg.condition]).unsqueeze(0).to(torch.device(arg.device))

    # predict
    hubert_feature = hubert_model(input_values_speech).last_hidden_state
    pic_feature = feature_net.net_vid.forward(emo_pic)
    pic_feature_double = feature_net.linear_interpolate_batch(pic_feature)

    if pic_feature_double.shape[1] < hubert_feature.shape[1]:
        while (pic_feature_double.shape[1] < hubert_feature.shape[1]):
            temp = pic_feature_double
            pic_feature_double = torch.cat([temp, temp], dim=1)
    pic_feature_double = pic_feature_double[:, :hubert_feature.shape[1], :]

    pr_video = model.predict(hubert_feature,pic_feature_double,person_id) #(1,time,53)
    # get vert
    bt=pr_video.shape[0]
    tim=pr_video.shape[1]
    exp = pr_video[:,:,:-3] #.to(torch.device(arg.device))
    jaw = pr_video[:,:,-3:] #.to(torch.device(arg.device))
    shape = torch.from_numpy(shape_dict[arg.condition]).unsqueeze(0)[:,None,:].repeat(1, tim, 1).to(torch.device(arg.device))
    rotation = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    neck_pose = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    eyes_pose = torch.zeros(bt,tim, 6).to(torch.device(arg.device))
    verts_video,ldmark= flame(shape,exp,rotation,neck_pose,jaw,eyes_pose)

    pred_video = verts_video[0]

    np.save(os.path.join(arg.resulllt_path, 'video_ref.npy'), pred_video.detach().cpu().numpy())

def main():
    parser = argparse.ArgumentParser(description='Ours model')
    parser.add_argument("--model_name", type=str, default="rav_hdtf")
    parser.add_argument("--fps", type=float, default=25, help='frame rate')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--audio_emo_path", type=str, default="demo/wav/audio_emo.wav", help='path of the input emotion audio')
    parser.add_argument("--video_emo_path", type=str, default="demo/wav/video_emo", help='path of the input emotion video pic')
    parser.add_argument("--result_path", type=str, default="demo/audio_video_driven_result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="Actor_01", help='select a conditioning subject from train_subjects')
    parser.add_argument("--ede_model_path", type=str, default="./checkpoint/EDE/model_ckpt_steps_7000.ckpt")
    parser.add_argument("--emg_model_path", type=str, default="./checkpoint/EMG/model_ckpt_steps_21000.ckpt")
    parser.add_argument("--person_dict_path", type=str, default="./cheeckpoint/person_dic.npz")
    parser.add_argument("--hubert_path", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--shape_dict_path", type=str, default="./cheeckpoint/shape_dic.npz")
    parser.add_argument("--driven_type", type=str, default="audio")
    args = parser.parse_args()   

    if args.driven_type == 'audio' :
        audio_driven(args)
    else:
        video_driven(args)

if __name__=="__main__":
    main()
