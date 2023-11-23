using Microsoft.ML;
using Microsoft.ML.Vision;
using System.Collections.Generic;
using Tranfer_Learning;
using Microsoft.ML.Transforms;
using System.Security.Cryptography;
using System;

// Initialize a new MLContext object
var mlContext = new MLContext();

// Path to the folder containing the image data
var folder = "C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\Tranfer Learning\\Dataset\\Train\\";
// Get a list of all files in the folder
var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);

// Initialize a list to store image data
IList<ImageData> images = new List<ImageData>();

// Iterate through all files and add them to the image list
foreach (var file in files)
{
    // Only process files with .jpg or .png format
    if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png")) continue;

    // The label of the image is determined by the name of the folder containing the image
    var label = Directory.GetParent(file).Name;
    images.Add(new ImageData { ImagePath = file, Label = label });
}

//// Create an IDataView object from the image list
//IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
//// Shuffle the data
//IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);

//// Create a pipeline to preprocess the data: convert labels to numerical values and load images as raw bytes
//var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
//    inputColumnName: "Label",
//    outputColumnName: "LabelAsKey")
//    .Append(mlContext.Transforms.LoadRawImageBytes(
//        outputColumnName: "Image",
//        imageFolder: ".",
//        inputColumnName: "ImagePath"
//        )
//    );

//// Preprocess the data
//var preprocessedData = pipeline.Fit(shuffledData).Transform(shuffledData);
//// Split the data into training and testing sets
//var trainTestData = mlContext.Data.TrainTestSplit(preprocessedData, testFraction: 0.2);

//// Continue preprocessing the data on the training set
//var transformedData = pipeline.Fit(trainTestData.TrainSet).Transform(trainTestData.TrainSet);

//// Set the options for the image classification model
//var options = new ImageClassificationTrainer.Options()
//{
//    FeatureColumnName = "Image",
//    LabelColumnName = "LabelAsKey",
//    ValidationSet = trainTestData.TestSet,
//    Arch = ImageClassificationTrainer.Architecture.ResnetV250,
//    MetricsCallback = (metrics) => Console.WriteLine(metrics),
//    TestOnTrainSet = false,
//    ReuseTrainSetBottleneckCachedValues = true,
//    ReuseValidationSetBottleneckCachedValues = true
//};

//// Add the model training step to the pipeline
//var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.ImageClassification(options))
//    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//// Train the model
//var trainedModel = trainingPipeline.Fit(trainTestData.TrainSet);

//// Predict on the test set
//var predictions = trainedModel.Transform(trainTestData.TestSet);

//// Evaluate the model
//var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");

//// Print out the evaluation metrics
//Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
//Console.WriteLine($"PerClassLogLoss is: {metrics.PerClassLogLoss}");

//// Save the model to a file
//mlContext.Model.Save(trainedModel, trainTestData.TrainSet.Schema, "model.zip");

// Load the model from the file
ITransformer loadedModel = mlContext.Model.Load("model.zip", out var modelSchema);

// Create a prediction engine from the loaded model
var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ModelOutput>(loadedModel);

// Use the prediction engine to predict the label of a new image
var singlePrediction = predictionEngine.Predict(new ImageData { ImagePath = "C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\Tranfer Learning\\Dataset\\Test\\Dog\\dog_168.jpg" });

// Print out the predicted label
Console.WriteLine($"Predicted label: {singlePrediction.PredictedLabel}");



