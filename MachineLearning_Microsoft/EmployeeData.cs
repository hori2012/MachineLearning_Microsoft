using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearning_Microsoft
{
    internal class EmployeeData
    {
        [LoadColumn(0)]
        public string? Education { get; set; }

        [LoadColumn(1)]
        public float JoiningYear { get; set; }

        [LoadColumn(2)]
        public string? City { get; set; }

        [LoadColumn(3)]
        public float PaymentTier { get; set; }

        [LoadColumn(4)]
        public float Age { get; set; }

        [LoadColumn(5)]
        public string? Gender { get; set; }

        [LoadColumn(6)]
        public bool EverBenched { get; set; }

        [LoadColumn(7)]
        public float ExperienceInCurrentDomain { get; set; }

        [LoadColumn(8)]
        public bool LeaveOrNot { get; set; }
    }
}
