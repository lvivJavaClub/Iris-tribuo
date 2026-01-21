# Iris Tribuo Example

This project demonstrates how to use the Iris dataset with Tribuo.

## Download and Setup

You can manually download and extract the data:
1. Download the data from: https://archive.ics.uci.edu/static/public/53/iris.zip
2. Extract the contents of the zip file into a folder named `data` in the project root.

Alternatively, use the following bash command:
```bash
  rm -rf data && mkdir -p data \
    && curl -L https://archive.ics.uci.edu/static/public/53/iris.zip -o data/iris.zip \
    && unzip data/iris.zip -d data && rm data/iris.zip
```

## Training the Model

To train the model, run the `TrainModel` class. This will load the Iris dataset, split it into training and testing sets, train a LibLinear model, evaluate its performance, and save the model to `models/model.tribuo`.

```bash
mvn exec:java -Dexec.mainClass="ua.lviv.javaclub.tribuo.training.TrainModel"
```

## Making Predictions

Once the model is trained and saved, you can use the `Predict` class to make predictions on new data.

```bash
mvn exec:java -Dexec.mainClass="ua.lviv.javaclub.tribuo.predict.Predict"
```