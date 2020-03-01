# Image Caption
## About
Inspired by Microsoft's CaptionBot project: 

https://www.captionbot.ai/

Here we try to replicate the results with a shallow LSTM network and a much smaller dataset for experimenting purpose. The obtained results were not perfect but quite acceptable considering that the computational power required is exponentially lower.

## Dataset

+ Flickr8k dataset (8.000 images)

https://www.kaggle.com/shadabhussain/flickr8k

Much larger datasets are also available for further improvement:

+ Flicker30k dataset (30.000 images)

https://www.kaggle.com/hsankesara/flickr-image-dataset

+ Microsoft's Coco dataset (180.000 images)

## Result

The model performs decently on images with similar contexts than thoses it has been trained on

![Screenshot from 2020-03-01 14-56-15](https://user-images.githubusercontent.com/37239826/75620726-b7e7c880-5bcf-11ea-84e8-6c43da4b5cf9.png)

The grammar is not exactly correct due to lemmatization

![Screenshot from 2020-03-01 15-02-40](https://user-images.githubusercontent.com/37239826/75620727-bd451300-5bcf-11ea-867a-1624604b0420.png)

The model fails to return a proper result because it doesn't have the word "computer games" in its vocabuary and therefore cannot understand the context of the image.

![Screenshot from 2020-03-01 15-05-25](https://user-images.githubusercontent.com/37239826/75620730-bfa76d00-5bcf-11ea-8c14-8a7c2244af9d.png)
