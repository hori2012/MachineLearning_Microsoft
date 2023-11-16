using Microsoft.ML;
using System.Collections.Generic;
using Tranfer_Learning;

var mlContext = new MLContext();
var folder = "C:\\Users\\Cong Hao\\Documents\\Data Mining\\ML.NET\\MachineLearning_Microsoft\\Tranfer Learning\\Dataset\\Train\\";
var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);

IList<ImageData> images = new List<ImageData>();

foreach (var file in files)
{
    if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png")) continue;

    var label = Directory.GetParent(file).Name;
    images.Add(new ImageData { ImagePath = file, Label = label });
}
IDataView imageData = mlContext.Data.LoadFromEnumerable(images);
IDataView shuffledData = mlContext.Data.ShuffleRows(imageData);


