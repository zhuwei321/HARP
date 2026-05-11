How to use？
Use git command to pull the project code:

gh repo clone zhuwei321/HARP
Download the dataset called basic_view_pn.csv and the video cover image compression package called img_yt.zip, unzip them and save them in the data_source folder. The google driver download disk link is:

https://drive.google.com/drive/folders/1PmUrmfCAyH-jzUP-BSk0KeEpx19nOaBM?usp=sharing
​ And downloaded baidu cloud download disk link is：

https://pan.baidu.com/s/1Uc9qv8O_1_Juh1xcf7hsdg?pwd=j8e2 
extract code: j8e2
​ Download the volume to the same directory and decompress it.

Set the file path of the dataset in the parser of main.py in the project code，and in smp_model.py, youtube_lstm3 is the model mentioned in our paper, and the paths of bert_model and token need to be set by yourself. In youtube_data_lstm in smp_data.py, set whether to use EP and the number of days you want to predict. **Note that the seq_len here needs to be consistent with the seq_len in main.py.**Then you can run this project:
nohup python main.py --train=True --K=0 > output.log 2>&1 &
If you want to test a trained model, please set the path of model in main.py. Then run:

python main.py --test=True --K=0 
