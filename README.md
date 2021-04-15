# MyTorch Processing

### Processing code for mytorch cloud machine learning framework.

## To train the model:

```
python train.py --network <network_name (alexnet,resnet,vgg etc)> --experiment <name_of_experiment> --classes class1,class2 --pretrained <PATH_TO_DATA>
```

Pass `--evaluate` flag to evaluate trained model

## Run the service

Note: you will have to install Flask `pip install Flask`

```
python app.py
```

## Test the api

```
POST http://localhost:5000/predict
```

body:

```
{
  image: <base64>
}
```
