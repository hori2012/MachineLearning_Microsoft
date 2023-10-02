using Microsoft.ML.Data;

internal class PredictEmployee
{
    [ColumnName("PredictLabel")]
    public bool LeaveOrNot { get; set; }
}