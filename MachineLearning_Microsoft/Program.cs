using MachineLearning_Microsoft;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.IO;
using Microsoft.ML.Data;

//Create object MLContext
MLContext mLContext = new MLContext();
//Load data into IDataview
IDataView dataView = mLContext.Data.LoadFromTextFile<EmployeeData>("C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\MachineLearning_Microsoft\\Employee.csv", separatorChar: ',', hasHeader: true);
// Split the data into training and testing sets
TrainTestData trainTestData = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
//Select decompression, special extraction for data
var pipeline = mLContext.Transforms.Categorical.OneHotEncoding(new[] {new InputOutputColumnPair("Education"),
        new InputOutputColumnPair("City"), new InputOutputColumnPair("Gender")})
    .Append(mLContext.Transforms.NormalizeMinMax("JoiningYear"))
    .Append(mLContext.Transforms.NormalizeMinMax("PaymentTier"))
    .Append(mLContext.Transforms.NormalizeMinMax("Age"))
    .Append(mLContext.Transforms.NormalizeMinMax("ExperienceInCurrentDomain"))
    .Append(mLContext.Transforms.Conversion.ConvertType("EverBenched", outputKind: DataKind.Single))
    .Append(mLContext.Transforms.Concatenate("Features", "Education", "JoiningYear", "City", "PaymentTier", "Age", "Gender", "EverBenched", "ExperienceInCurrentDomain"))
    .Append(mLContext.BinaryClassification.Trainers.LdSvm(labelColumnName: "LeaveOrNot", featureColumnName: "Features"))
    .Append(mLContext.BinaryClassification.Calibrators.Platt(labelColumnName: "LeaveOrNot"));
//Train model
var model = pipeline.Fit(trainTestData.TrainSet);
//Metrics model
var metrics = mLContext.BinaryClassification.Evaluate(model.Transform(trainTestData.TestSet), labelColumnName: "LeaveOrNot");
//Do chinh xac
Console.WriteLine($"Accuracy: {metrics.Accuracy}");
//Cang gan 1 cang tot
Console.WriteLine($"F1 Score: {metrics.F1Score}");
//Cang thap cang tot
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
//Cang ve 0 cang tot
Console.WriteLine($"Entropy: {metrics.Entropy}");
//Save model
mLContext.Model.Save(model, trainTestData.TrainSet.Schema, "modelEmployee.zip");
//Using model is save
DataViewSchema modelSchema;
var modelSave = mLContext.Model.Load("modelEmployee.zip", out modelSchema);
var sampleData = new EmployeeData { Education = "Bachelors", JoiningYear = 2016, City = "Bangalore", PaymentTier = 1, Age = 33, Gender = "Female", EverBenched = false, ExperienceInCurrentDomain = 0 };
var output = mLContext.Model.CreatePredictionEngine<EmployeeData, PredictEmployee>(modelSave).Predict(sampleData);
Console.WriteLine("Result: " + output.LeaveOrNot);