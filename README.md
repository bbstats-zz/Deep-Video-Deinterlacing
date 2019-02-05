# Deep Video Deinterlacing

A fork for easy use on video files rather than single images, by streaming ffmpeg > tensorflor > ffmpeg. Use tensorflow-gpu or this will take forever on your cpu.

You need the following to run this:
- python 3.6
- numpy
- scipy
- tensorflow (gpu version is preferable!!!)
- ffmpeg-python
- matplotlib

To run, open a command line in the folder containing the scripts and type:

`python deinterlace.py [INPUT_VIDEO] [OUTPUT_VIDEO]`
eg.
`python deinterlace.py "test.ts" "output.mp4"`

Below is the original readme:


We run this code using Tensorflow.

### Architecture
TensorFlow Implementation of ["Real-time Deep Video Deinterlacing"](https://arxiv.org/abs/1708.00187)

<a href="http://tensorlayer.readthedocs.io">
<div align="center">
	<img src="images/model.png" width="100%"/>
</div>
</a>

### Results
<div align="center">
	<img src="images/10064.png" width="50%"/>
</div>
<div align="center">
	<img src="results/10064_0.png" width="80%"/>
</div>
<div align="center">
	<img src="results/10064_1.png" width="80%"/>
</div>

### Run
- Start deinterlacing
```
python runDeinterlacing.py --img_path=images/4.png
``` 

### Author
- [lszhuhaichao](https://github.com/lszhuhaichao)

### Citation
If you find this project useful, we will be grateful if you cite our paper

```
@article{zhu2017real,
  title={Real-time Deep Video Deinterlacing},
  author={Zhu, Haichao and Liu, Xueting and Mao, Xiangyu and Wong, Tien-Tsin},
  journal={arXiv preprint arXiv:1708.00187},
  year={2017}
}
```

### License
- For academic and non-commercial use only.
