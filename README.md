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


The server should be able to run, using the script server.py at the root of this repo, as it follows:


```
python3 server.py
```




Now you can access it at: http://0.0.0.0:5000
