# PatternReasoning



## How to install / launch


Clone this repo and install Python dependecies as it follows (you may want to use a Vitural env for that):


```
pip install -r requirements.txt
```

Then, you must copy the model downloaded from the given Google Drive into "model/src/pretrain":

```
mv ~/Downloads/lxmert.pth model/src/pretrain/lxmert.pth
```

And the TSV, also in the drive:

```
mkdir -p model/gqa_testdev_obj36/vg_gqa_imgfeat; mv ~/Downloads/gqa_testdev_obj36.tsv $_
```

This command creates non-existant Dirs, and moves the tsv, you may need to change it based on your OS (in linux **$_** at the end is needed)



You can launch the server with the script 'server.py' at the root of this repo:


```
python3 server.py
```




The server is then accessible at: http://0.0.0.0:5000 (it may take a minute or two to launch).
