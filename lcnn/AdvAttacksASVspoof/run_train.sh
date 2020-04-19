

configPath1=/content/AdvAttacksASVspoof/_configs/config_LA_lcnnHalf_LPSseg_uf_seg600.json
GPU1=0

python /content/AdvAttacksASVspoof/train.py --config ${configPath1} --device ${GPU1} 1> output_18_4_1.log 2>&1 #> /dev/null 2>&1 &



# ps -ef  | grep python
# kill -9 ps_id
