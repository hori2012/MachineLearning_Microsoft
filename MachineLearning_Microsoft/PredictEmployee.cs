using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning_Microsoft
{
    internal class PredictEmployee
    {
        [ColumnName("PredictLabel")]
        public bool LeaveOrNot { get; set; }
    }
}
