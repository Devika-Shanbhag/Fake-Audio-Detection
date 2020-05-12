

configPath1=_configs/config_LA_lcnnHalf_LPSseg_uf_seg600.json
GPU1=0

python train.py --config ${configPath1} --device ${GPU1} 1> output_1_5_2020_2.log #> /dev/null 2>&1 &



# ps -ef  | grep python
# kill -9 ps_id
