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

def test_model(arg):
    if not os.path.exists(arg.result_path):
        os.makedirs(arg.result_path)
    #subj_list
    person_dic = np.load(arg.person_dict_path)
    person_dict = {key: value for key, value in person_dic.items()}
    #shape_list
    shape_dic = np.load(arg.shape_dict_path)
    shape_dict = {key: value for key, value in shape_dic.items()}
    #avcc_hubert
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
    print("------loading emg model-------")
    model = VAEModel(in_out_dim=53)
    ckp=torch.load(arg.emg_model_path)['state_dict']['model']
    model.load_state_dict(ckp) # path of the vggish pretrained network
    model = model.to(torch.device(arg.device))
    model.eval()
    #flame model
    from flame_model.flame import FlameHead
    from flame_model.render import mesh_render,render_sequence_meshes
    flame=FlameHead(50,100,add_teeth=False).to(torch.device(arg.device))
    flame.eval()
    #prepare input data
    wav_path = arg.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    input_values_all = processor(speech_array, return_tensors="pt", sampling_rate=16000).input_values # [1, T]
    input_values_all = input_values_all.to(torch.device(arg.device))

    person_id = torch.tensor(person_dict[arg.condition]).unsqueeze(0).to(torch.device(arg.device))
    #split long audio
    count= 1
    pred = []
    print('------start predict------')
    start_time = time.time()

    hubert_feature = hubert_model(input_values_all).last_hidden_state
    emo_feature = feature_net.net_aud.forward(input_values_all).last_hidden_state
    pr = model.predict(hubert_feature,emo_feature,person_id) #(1,time,53)

    #get vert
    bt=pr.shape[0]
    tim=pr.shape[1]
    exp = pr[:,:,:-3] #.to(torch.device(arg.device))
    jaw = pr[:,:,-3:] #.to(torch.device(arg.device))
    shape = torch.from_numpy(shape_dict[arg.condition]).unsqueeze(0)[:,None,:].repeat(1, tim, 1).to(torch.device(arg.device))
    rotation = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    neck_pose = torch.zeros(bt,tim, 3).to(torch.device(arg.device))
    eyes_pose = torch.zeros(bt,tim, 6).to(torch.device(arg.device))
    pred,ldmark= flame(shape,exp,rotation,neck_pose,jaw,eyes_pose)

    end_time = time.time()
    time_use = end_time - start_time
    fps = pred.shape[1] / time_use

    pred = pred[0]
    print(pred.shape)
    np.save(os.path.join(arg.result_path, test_name+'.npy'), pred.detach().cpu().numpy())
    print(f"successfully saved {os.path.join(arg.result_path, test_name+'.npy')}, use {time_use} s, fps {fps}")

def main():
    parser = argparse.ArgumentParser(description='Ours model')
    parser.add_argument("--model_name", type=str, default="rav_hdtf")
    parser.add_argument("--fps", type=float, default=25, help='frame rate')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="Actor_01", help='select a conditioning subject from train_subjects')
    parser.add_argument("--ede_model_path", type=str, default="./checkpoint/EDE/model_ckpt_steps_7000.ckpt")
    parser.add_argument("--emg_model_path", type=str, default="./checkpoint/EMG/model_ckpt_steps_21000.ckpt")
    parser.add_argument("--person_dict_path", type=str, default="./checkpoint/person_dic.npz")
    parser.add_argument("--hubert_path", type=str, default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--shape_dict_path", type=str, default="./checkpoint/shape_dic.npz")
    args = parser.parse_args()   
    
    test_model(args)

if __name__=="__main__":
    main()
