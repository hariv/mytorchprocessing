MyTorch Processing
==================

### Processing code for mytorch cloud machine learning framework.

All code is present inside the classifier directory. Ignore everything else.

Currently, the code supports different models for classifiying classes on the 
Imagenet dataset (http://www.image-net.org/). It will soon support other
classification tasks such as traffic sign classification etc. It will
also subsequently support training of different models.

## To test the model.

```
python test.py --dataroot <path_to_test_dataset> --name <name_of_experiment> --model <name_of_model_to_be_used>
```

Further options can be seen in options/test_options.py