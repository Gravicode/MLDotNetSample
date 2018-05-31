using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace TaxiFarePredictor
{
    class Program
    {
        static void Main(string[] args)
        {
            var DataDir = @"C:\experiment\MLDotNet\MLDotNetSample\Data\";
          
            //get data for training
            string DataPath = DataDir+ "taxi-fare-train.csv";
            var pipeline = new LearningPipeline();
            //load file
            pipeline.Add(new TextLoader<TaxiTrip>(DataPath, useHeader: true, separator: ","));
            //add numerical column
            pipeline.Add(new ColumnConcatenator("NumericalFeatures", "passenger_count", "trip_time_in_secs", "trip_distance"));
            //add categorical column 
            pipeline.Add(new ColumnConcatenator("CategoryFeatures", "vendor_id", "rate_code", "payment_type"));
            pipeline.Add(new CategoricalOneHotVectorizer("CategoryFeatures"));
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "NumericalFeatures", "CategoryFeatures"));
            //set algorthm
            pipeline.Add(new FastTreeRegressor());
            //Training
            var model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            var testData = new TextLoader<TaxiTrip>(DataDir+"taxi-fare-test.csv", useHeader: true);

            //Test
            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine("Root Mean Square is {0}", metrics.Rms);

            //Predict
            TaxiTripFarePrediction prediction = model.Predict(new TaxiTrip()
            {
                passenger_count = 1,
                payment_type = "CRD",
                rate_code = "1",
                trip_distance = 3.75f,
                trip_time_in_secs = 1140,
                vendor_id = "VTS"

            });
            Console.WriteLine("Prediksi:" + prediction.fare_amount);

        }
    }
    class TaxiTrip
    {

        [Column(ordinal: "0")]
        public string vendor_id;
        [Column(ordinal: "1")]
        public string rate_code;
        [Column(ordinal: "2")]
        public float passenger_count;
        [Column(ordinal: "3")]
        public float trip_time_in_secs;
        [Column(ordinal: "4")]
        public float trip_distance;
        [Column(ordinal: "5")]
        public string payment_type;
        [Column(ordinal: "6", name: "Label")]
        public float fare_amount;
    }
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float fare_amount;
    }
}
