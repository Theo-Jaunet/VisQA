# VisQA: X-raying Vision and Language Reasoning in Transformers

VisQA is an instance-based interactive visualization system designed to help Visual Question Answering (VQA) practitioners to analyze attention maps in transformer models, prune them, and ask free-form questions, in order to grasp their tendency to exploit shortcuts from their training to provide answers.


<p align="center">
<img src="https://github.com/Theo-Jaunet/VisQA/blob/master/teaserhd.jpg" height="450">
 <p align="center">
 Opening the black box of neural models for vision and language reasoning: given an open-ended question and an image ①, VisQA enables to investigate whether a trained model resorts to reasoning or to bias exploitation to provide its answer. This can be achieved by exploring the behavior of a set of attention heads ②, each producing an attention map ⑤, which manage how different items of the problem relate to each other. Heads can be selected ③, for instance, based on color-coded activity statistics. Their semantics can be linked to language functions derived from dataset-level statistics ④, filtered and compared between different models.
  </p>
</p>



For more information, please refer to the manuscript: 
[VisQA: X-raying Vision and Language Reasoning in Transformers](https://arxiv.org/abs/2104.00926)

Work by:  Theo Jaunet*, Corentin Kervadec*, Romain Vuillemot, Grigory Antipov, Moez Baccouche, and Christian Wolf

<em> (\*) indicates equal contribution </em>



## Live Demo
(Designed to work on Google Chrome, at 1920*1080)

This tool is accessible online using the following link: https://visqa.liris.cnrs.fr/


## How to install locally

Clone this repo and install Python dependecies as it follows (you may want to use a vitural environment for that):


```
pip install -r requirements.txt
```

For the following, please refer to LXMERT's repo : 
 https://github.com/airsplay/lxmert
 
 
You must add models into "model/src/pretrain", for instance, the following:

```
mv ~/Downloads/lxmert.pth model/src/pretrain/lxmert.pth

mv ~/Downloads/oracle.pth model/src/pretrain/oracle.pth

mv ~/Downloads/tiny_lxmert.pth model/src/pretrain/tiny_lxmert.pth
```

And the features TSV:

```
mkdir -p model/gqa_testdev_obj36/vg_gqa_imgfeat; mv ~/Downloads/gqa_testdev_obj36.tsv $_
```

This command creates non-existant Dirs, and moves the tsv, you may need to change it based on your OS (in linux **$_** at the end is needed)


You can launch the server with the script 'visqa.py' at the root of this repo:


```
python visqa.py
```



The server is then accessible at: http://0.0.0.0:5000 (it may take a minute or two to launch).


## Citation

If you find this usefull, consider citing the following:

```
@inproceedings{Jaunet2021VisQA,
    author = {Theo Jaunet, Corentin Kervadec, Romain Vuillemot, Grigory Antipov, Moez Baccouche, and Christian Wolf},
    title = {VisQA: X-raying Vision and Language Reasoning in Transformers},
    journal={IEEE Transactions on Visualization and Computer Graphics (TVCG)},
    year={2021},
    publisher={IEEE}
}
```



