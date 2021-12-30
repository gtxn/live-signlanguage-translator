# live-signlanguage-translator

This project made use of @mediapipe as well as tensorflow to quickly and accurately identify letters in the American Sign Language.

I made use of the following kaggle dataset
➡️ https://www.kaggle.com/grassknoted/asl-alphabet

## Try it yourself
```pip3 install -r requirements.txt```

```python example.py```

Note: It may take awhile to load the model so patience is key!

If it doesn't work, try changing line thirty to code below:
```cap = cv2.VideoCapture(0)```

## Initial processing
First I passed each image through mediapipe's hand detection model, which gave 21 (x, y, z) coordinates of the different hand landmarks. 

But to make my model more accurate, I did additional processing to calculate relative distances away from landmark 0 (the wrist) instead of the raw values themselves.

## Training time
Next, I processed the data as an Numpy array and fed it into a Tensorflow neural network. And trained it on 500 epochs.

![image](https://user-images.githubusercontent.com/59089164/147747344-d59c7adf-3465-4566-9b52-e6bd32d4bfaa.png)

The end result was pretty great! The training accuracy was 99.51% and evaluation on test data produced accuracy of 94.91%.

## Final product
Using cv2, I ran @mediapipe's hand detection on each frame and subsequently passed the coordinates to my model to allow live translation of American Sign Language. 

## Random thoughts 🧠
Although I doubt this is the best way to train a model for sign language, I can see how leveraging on a pretrained hand landmark detection model could be useful especially for more complex actions. But overall, fun project weehoo!
