import subprocess
from pathlib import Path


root_path = Path('/mnt/data1/input/SAR-RARP50/20220723/videos/')
input_dirs = root_path.glob('*/rgb_jpg')
flow_type = 'tvl1'

for input_dir in input_dirs:
    output_dir = input_dir.parent / f'flow_{flow_type}'
    subprocess.call(f'python build_rawframes.py {input_dir} {output_dir} --task flow --input-frames --flow-type {flow_type} --out-format jpg --level 0 --report-file build_report.txt', shell=True)
