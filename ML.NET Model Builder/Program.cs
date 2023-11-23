using ML_NET_Model_Builder;

// Create single instance of sample data from first line of dataset for model input
var imageBytes = File.ReadAllBytes(@"C:\Users\Cong Hao\Documents\Data Mining\ML.NET\MachineLearning_Microsoft\ML.NET Model Builder\Images\ImageTest\Dogs\202.jpg");

ClassficationImages.ModelInput sampleData = new ClassficationImages.ModelInput()
{
    ImageSource = imageBytes,
};
// Make a single prediction on the sample data and print results.
var sortedScoresWithLabel = ClassficationImages.PredictAllLabels(sampleData);
Console.WriteLine($"{"Class",-40}{"Score",-20}");
Console.WriteLine($"{"-----",-40}{"-----",-20}");

foreach (var score in sortedScoresWithLabel)
{
    Console.WriteLine($"{score.Key,-40}{score.Value,-20}");
}