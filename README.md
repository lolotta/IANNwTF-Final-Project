# IANNwTF-Final-Project


## Project:

Single-Image Colorization (estimating RGB colors for grayscale images)

Train CNN on colorization and classification task

How does colorization affect classification and vice versa?

Does colorization profit from high-level concepts like body parts, background, etc. (as this would theoretically also help with classification)? 


## Dataset:
The Stanford Dog Dataset can be found here: 
http://vision.stanford.edu/aditya86/ImageNetDogs/ 

Number of categories (dog breeds): 120
Number of images: 20.580

Images can be downloaded via this link: http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar 


## Architecture: 
![architecture](https://user-images.githubusercontent.com/46372060/223740729-c5bd76aa-8112-4910-9c81-14c922ac47a2.jpg)
(Iizuka et al., 2016)


## Work Log:

| When  | Who    |  What |
|:------|:-------|:------:|
|20-23 Feb | all | brainstorming ideas, finalizing project task, search for papers/code inspiration |
|23rd Feb | Lotta |data preprocessing script, first code draft of model architecture (see picture above) |
|24th Feb |Lotta | first training of model with fusion layer → loss is giving unclear results |
|27th Feb |Lotta | inspection of the colorized images of trained model →  mostly looking beige; model does not seem to properly learn the coloriz. task |
|1st March | Lotta |working on code; new inspection of results → unrealistic accuracy results for classification (problem with labels?); weirdly colored images (problem lies in tfio.experimental-color.rgb_to_lab() function) |
|2nd March |Lotta | fixed colorization issue; now training the two models for the two individual tasks → results now look like the model learns a beige filter; loss for coloriz. task looks ok; loss for classif. task looks like overfitting (and does not learn the dog breeds) |
| | Lotta | training the big joint model (coloriz. and classif. task) → coloriz. similar to before (single task model) and classif. still bad |
|3rd March | Lotta | to improve classification, train model with additional dropout layer, maxpool layers and Adam optimizer instead of Adadelta → coloriz. similar to before; classif. accuracy still bad; brainstorming ideas how to improve classif. performance (see below) |
|8th March | all, Leon | first project meeting with tutor 


### Further ideas: 

* to improve classification try easier dataset (images will less background noise to make task easier) 
* use the colorization dataset ‘Natural-Color dataset (NCD)’ from Anwar et al. (2020) to test how good model colorizes on a dataset specifically made for the task (color groundtruth: https://drive.google.com/file/d/1k_UvYzdrHbphW4UcbDb9jWB0ZQIAGEAo/view ; greyscale: https://drive.google.com/file/d/1GpmEVNFn12bK0EoXK46FP3cXFUosomaG/view )
* consider what color space to use: RGB, YUV vs. L* a * b * (CIELAB)
* consider loss function (see Huang et al., 2022) 
* try different model architecture where we can add pre-trained model like e.g. Inception-ResNet-v2 (see Baldassarre et al., 2017)


## Big To-Do’s:

* improve model performance
* clean up and comment code
* add requirements.txt
* write report


## Literature:

1. Baldassarre, F., Morín, D. G., & Rodés-Guirao, L. (2017). Deep koalarization: Image colorization using cnns and inception-resnet-v2. arXiv preprint arXiv:1712.03400.

2. Iizuka, S., Simo-Serra, E., & Ishikawa, H. (2016). Let there be color! Joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics (ToG), 35(4), 1-11.

https://github.com/satoshiiizuka/siggraph2016_colorization 

3. Anwar, S., Tahir, M., Li, C., Mian, A., Khan, F. S., & Muzaffar, A. W. (2020). Image colorization: A survey and dataset. arXiv preprint arXiv:2008.10774.

https://github.com/saeed-anwar/ColorSurvey 

4. Huang, S., Jin, X., Jiang, Q., & Liu, L. (2022). Deep learning for image colorization: Current and future prospects. Engineering Applications of Artificial Intelligence, 114, 105006.


### More on CNN architecture:

Bhatt D., Patel C., Talsania H., Patel J., Vaghela R., Pandya S., Modi K., Ghayvat H. (2021). CNN Variants for Computer Vision: History, Architecture, Application, Challenges and Future Scope. Electronics 10(20):2470. https://doi.org/10.3390/electronics10202470 

Ajit, A., Acharya K., & Samanta, A. (2020). A Review of Convolutional Neural Networks. 2020 International Conference on Emerging Trends in Information Technology and Engineering (ic-ETITE), pp. 1-5, https://doi.org/10.1109/ic-ETITE47903.2020.049   

