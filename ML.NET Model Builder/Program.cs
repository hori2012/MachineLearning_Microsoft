//Load sample data
using ML_NET_Model_Builder;

var imageBytes = File.ReadAllBytes(@"C:\Users\Cong Hao\Documents\Data Mining\ML.NET\MachineLearning_Microsoft\ML.NET Model Builder\Images\ImageTest\Dogs\202.jpg");
ClassficationImages.ModelInput sampleData = new ClassficationImages.ModelInput()
{
    ImageSource = imageBytes,
};

//Load model and predict output
var result = ClassficationImages.Predict(sampleData);

Console.WriteLine($"ImageSource: {result.ImageSource}");
Console.WriteLine($"Label: {result.Label}");
Console.WriteLine($"PredictedLabel: {result.PredictedLabel}");
Console.WriteLine($"Score: {string.Join(" , ", result.Score)}");
