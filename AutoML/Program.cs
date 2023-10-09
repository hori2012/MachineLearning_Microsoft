using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;

MLContext mLContext = new MLContext();
//Infer column information
ColumnInferenceResults columnInference = mLContext.Auto().InferColumns("C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\AutoML\\Employee.csv", labelColumnName: "LeaveOrNot", groupColumns: false);
//Create text loader
TextLoader textLoader = mLContext.Data.CreateTextLoader(columnInference.TextLoaderOptions);
//load data into IDateView
IDataView dataView = textLoader.Load("C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\AutoML\\Employee.csv");
//Split the data into training and testing sets
TrainTestData trainTestData = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

//Define the experiment settings
var experimentSettings = new BinaryExperimentSettings();
experimentSettings.MaxExperimentTimeInSeconds = 400;
//Create the experiment
var experiment = mLContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);

//Run the experiment on the training data and get the best model
var result = experiment.Execute(trainTestData.TrainSet, columnInference.ColumnInformation);

//Get the best model
ITransformer model = result.BestRun.Model;
var metrics = mLContext.BinaryClassification.Evaluate(model.Transform(trainTestData.TestSet), labelColumnName: "LeaveOrNot");
Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"F1 Score: {metrics.F1Score}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine($"Entropy: {metrics.Entropy}");
//Save model
mLContext.Model.Save(model, trainTestData.TrainSet.Schema, "modelEmployee.zip");
//Load model save
DataViewSchema modelSchema;
var modelSave = mLContext.Model.Load("modelEmployee.zip", out modelSchema);
var sampleData = new EmployeeData { Education = "Masters", JoiningYear = 2018, City = "New Delhi", PaymentTier = 3, Age = 27, Gender = "Male", EverBenched = false, ExperienceInCurrentDomain = 5 };
var output = mLContext.Model.CreatePredictionEngine<EmployeeData, PredictEmployee>(modelSave).Predict(sampleData);
Console.WriteLine("Result: " + output.LeaveOrNot);