Test script for Track s7:

**Note:** Test data should be under the folder named "data/s7/test_rgb". Checkpoints should be under the folder named "ckpts/s7". 

```
python main.py --data_path data/s7/ --save_dir ckpts/s7/ --output_dir outputs/ --ckpt_path ckpts/s7/checkpoint-best.pth
```


Test script for Track p20:

**Note:** Test data should be under the folder named "data/p20/test_rgb". Checkpoints should be under the folder named "ckpts/p20". 

```
python main.py --data_path data/p20/ --save_dir ckpts/p20/ --output_dir outputs/ --ckpt_path ckpts/p20/checkpoint-best.pth
```