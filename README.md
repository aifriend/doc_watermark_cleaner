# DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement

## Description

This is an implementation for the
paper [DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement](https://ieeexplore.ieee.org/document/9187695)<br>
DE-GAN is a conditional generative adversarial network designed to enhance the document quality before the recognition
process. It could be used for document cleaning, binarization, deblurring and watermark removal. The weights are
available to test the enhancement.

## License

This work is only allowed for academic research use. For commercial use, please contact the author.

## Download

- Clone this repo:

```bash
git clone https://github.com/dali92002/DE-GAN
cd DE-GAN
```

- Then, download the trained weghts to directly use the model for document enhancement, it is important to save these
  weights in the subfolder named weights, in the DE-GAN folder. The link to download the weights
  is : https://drive.google.com/file/d/1J_t-TzR2rxp94SzfPoeuJniSFLfY3HM-/view?usp=sharing

## Requirements

- install the requirements.txt

## Using DE-GAN

### Document binarization

- To binarize an image use the followng command:

```bash
python enhance.py binarize ./image_to_binarize ./directory_to_binarized_image
```

image:<br />
![img.png](src/img.png)

binarized image:<br />
![img_1.png](src/img_1.png)

### Document deblurring

- To deblur an image use the followng command:

```bash
python enhance.py deblur ./image_to_deblur ./directory_to_deblurred_image
```

blurred image:<br />
![img_2.png](src/img_2.png)

enhanced image:<br />
![img_3.png](src/img_3.png)

### Watermark removal

- To remove a watermark from an image use the followng command:

```bash
python enhance.py unwatermark ./image_to_unwatermark ./directory_to_unwatermarked_image
```

watermarked image:<br />
![img_4.png](src/img_4.png)

clean image:<br />
![img_5.png](src/img_5.png)

### Document cleaning

degraded image:<br />
![img_6.png](src/img_6.png)

cleaned image:<br />
![img_7.png](src/img_7.png)

## Training with your own data

- To train with your own data, place your degraded images in the folder "path_to/wm/" and the corresponding ground-truth
  in the folder "path_to/gt/". It is necessary that each degraded image and its corresponding gt are having the same
  name (could have different extentions), also, the number images should be the same in both folders.
- Command to train:

```bash
python train.py 
```

- Specifying the batch size and the number of epochs could be done inside the code.

## Citation

- If this work was useful for you, please cite it as:

```
@ARTICLE{Souibgui2020,
  author={Mohamed Ali Souibgui  and Yousri Kessentini},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={DE-GAN: A Conditional Generative Adversarial Network for Document Enhancement}, 
  year={2020},
  doi={10.1109/TPAMI.2020.3022406}}
```