#data for idx
sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/idx/sbp_position_min0.15_max0.85.npy"
time_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/idx/timestamps_position_min0.15_max0.85.npy"

#data for mrs
# sbp_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/sbp_position_min0.15_max0.85.npy"
# time_info_path="/Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/Data/mrs/timestamps_position_min0.15_max0.85.npy"

python -m src.RL_decoders.position_decoder --sbp_path ${sbp_path} --time_info_path ${time_info_path}

#original: 
# python -m RL_decoders.RL-decoders-iBMI \
#     --dir /Users/chanyu/Dropbox/NeuroData2025/BIU/ML_proj/RLINK/src/RL_decoders/datasets/derived-RL-expt \
#     --expt monkey_1_set_1