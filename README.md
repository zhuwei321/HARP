# **HARP: Retrieval-Augmented Temporal Popularity Prediction forSocial Media Content** 

<img src="https://github.com/zhuwei321/HARP/blob/main/images/model.png" style="zoom: 15%;" />
<img src="https://github.com/zhuwei321/HARP/blob/main/images/model2.png" style="zoom: 15%;" />

## About

​Predicting the popularity of social media content is crucial for applications such as personalized recommendation and viral trend analysis.
However, existing datasets and methods often rely on isolated popularity scores or target-side multimodal cues, making it difficult to evaluate temporally aligned popularity evolution and to transfer propagation patterns from related historical content. 
To address these challenges, we build upon the Social Media Temporal Popularity Prediction Dataset~(SMTPD), a multilingual multimodal benchmark with 30-day aligned popularity trajectories, and propose HARP~(Hybrid Aggregation for Recurrent Popularity Modeling). 
HARP first estimates a target-conditioned preliminary popularity trajectory from publication-time cues and then uses it as a popularity-aware query for hybrid retrieval.
Specifically, a graph-guided multimodal hybrid retrieval aggregation module selects top-$k$ historical references by combining trajectory similarity with hypergraph-based semantic structure, and aggregates visual, textual, frequency-domain, and early-popularity features as neighborhood priors. 
These retrieval-enhanced representations are further integrated for temporally aligned popularity forecasting.
Experimental results on SMTPD show that HARP consistently outperforms representative baselines, demonstrating the effectiveness of popularity-aware retrieval and multimodal neighborhood aggregation for social media popularity prediction.

## How to use？

1. Use git command to pull the project code:

   ```
   gh repo clone zhuwei321/HARP
   ```

2. Download the dataset called basic_view_pn.csv and the video cover image compression package called img_yt.zip, unzip them and save them in the data_source folder.
    The  google driver download disk link is:

  ```
  https://drive.google.com/drive/folders/1PmUrmfCAyH-jzUP-BSk0KeEpx19nOaBM?usp=sharing
  ```

​        And downloaded baidu cloud download disk link is：

```
https://pan.baidu.com/s/1Uc9qv8O_1_Juh1xcf7hsdg?pwd=j8e2 
extract code: j8e2
```

​       Download the  volume to the same directory and decompress it.

3. Set the file path of the dataset in the parser of main.py in the project code，and in smp_model.py,   youtube_lstm3 is the model mentioned in our paper, and the paths of bert_model and token need to be set by yourself. In youtube_data_lstm in smp_data.py, set whether to use EP and the number of days you want to predict. **Note that the seq_len here needs to be consistent with the seq_len in main.py.**Then you can run this project:

```
nohup python main.py --train=True --K=0 > output.log 2>&1 &
```

If you want to test a trained model, please set the path of model in main.py. Then run:

```
python main.py --test=True --K=0 
```
