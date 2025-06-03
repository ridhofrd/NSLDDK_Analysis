#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Struktur untuk menyimpan data
struct DataRecord
{
    vector<double> features;
    string label;
};

struct GroupedData
{
    int no;
    double startRange;
    double endRange;
    int frequency;
};

// Struktur untuk hasil analisis statistik
struct Statistics
{
    double mean;
    double median;
    double mode;
    double stddev;
    double variance;
    double min;
    double max;
    double skewness;
    double kurtosis;
};

// Struktur untuk hasil korelasi
struct CorrelationResult
{
    string attribute;
    double correlation;
    int index;
};

class KDDAnalyzer
{
private:
    vector<DataRecord> trainData;
    vector<DataRecord> testData;
    vector<string> attributeNames;
    vector<int> realAttributeIndices;
    vector<GroupedData> groupedData;

public:
    // Fungsi untuk membaca file ARFF
    void readARFF(const string &filename, vector<DataRecord> &data)
    {
        ifstream file(filename);
        string line;
        bool dataSection = false;

        attributeNames.clear();
        realAttributeIndices.clear();

        while (getline(file, line))
        {
            // Skip empty lines
            if (line.empty())
                continue;

            // Parse attribute names and types
            if (line.find("@attribute") != string::npos)
            {
                size_t start = line.find("'") + 1;
                size_t end = line.find("'", start);
                string attrName = line.substr(start, end - start);
                attributeNames.push_back(attrName);

                // Check if it's a real attribute
                if (line.find(" real") != string::npos)
                {
                    realAttributeIndices.push_back(attributeNames.size() - 1);
                }
            }

            // Start reading data
            if (line.find("@data") != string::npos)
            {
                dataSection = true;
                continue;
            }

            // Parse data records
            if (dataSection)
            {
                DataRecord record;
                stringstream ss(line);
                string value;
                int index = 0;

                while (getline(ss, value, ','))
                {
                    if (index < attributeNames.size() - 1)
                    {
                        try
                        {
                            record.features.push_back(stod(value));
                        }
                        catch (...)
                        {
                            record.features.push_back(0.0); // Handle non-numeric values
                        }
                    }
                    else
                    {
                        record.label = value;
                    }
                    index++;
                }

                if (!record.features.empty())
                {
                    data.push_back(record);
                }
            }
        }
        file.close();
    }

    // Fungsi untuk menghitung korelasi Pearson
    double calculatePearsonCorrelation(const vector<double> &x, const vector<double> &y)
    {
        if (x.size() != y.size() || x.empty())
            return 0.0;

        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
        int n = x.size();

        for (int i = 0; i < n; i++)
        {
            sumX += x[i];
            sumY += y[i];
            sumXY += x[i] * y[i];
            sumX2 += x[i] * x[i];
            sumY2 += y[i] * y[i];
        }

        double num = n * sumXY - sumX * sumY;
        double den = sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

        if (den == 0)
            return 0.0;
        return num / den;
    }

    // Fungsi untuk analisis korelasi
    vector<CorrelationResult> performCorrelationAnalysis(const vector<DataRecord> &data)
    {
        vector<CorrelationResult> results;

        // Convert labels to binary (0 for normal, 1 for anomaly)
        vector<double> labels;
        for (const auto &record : data)
        {
            labels.push_back(record.label == "anomaly" ? 1.0 : 0.0);
        }

        // Calculate correlation for each real attribute
        for (int idx : realAttributeIndices)
        {
            vector<double> feature;
            for (const auto &record : data)
            {
                if (idx < record.features.size())
                {
                    feature.push_back(record.features[idx]);
                }
            }

            double corr = calculatePearsonCorrelation(feature, labels);
            results.push_back({attributeNames[idx], abs(corr), idx});
        }

        // Sort by absolute correlation value
        sort(results.begin(), results.end(),
             [](const CorrelationResult &a, const CorrelationResult &b)
             {
                 return a.correlation > b.correlation;
             });

        return results;
    }

    // Fungsi untuk menghitung statistik deskriptif
    Statistics calculateStatistics(const vector<double> &data)
    {
        Statistics stats;

        if (data.empty())
            return stats;

        // Mean
        stats.mean = accumulate(data.begin(), data.end(), 0.0) / data.size();

        // Sort for median and mode
        vector<double> sorted = data;
        sort(sorted.begin(), sorted.end());

        // Median
        int n = sorted.size();
        if (n % 2 == 0)
        {
            stats.median = (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }
        else
        {
            stats.median = sorted[n / 2];
        }

        // Mode (simplified - most frequent value)
        map<int, int> frequency;
        for (double val : data)
        {
            frequency[(int)(val * 100)]++; // Group by 2 decimal places
        }
        int maxFreq = 0;
        stats.mode = 0;
        for (const auto &pair : frequency)
        {
            if (pair.second > maxFreq)
            {
                maxFreq = pair.second;
                stats.mode = pair.first / 100.0;
            }
        }

        // Min and Max
        stats.min = *min_element(data.begin(), data.end());
        stats.max = *max_element(data.begin(), data.end());

        // Variance and Standard Deviation
        double sumSquaredDiff = 0;
        for (double val : data)
        {
            sumSquaredDiff += pow(val - stats.mean, 2);
        }
        stats.variance = sumSquaredDiff / data.size();
        stats.stddev = sqrt(stats.variance);

        // Skewness
        double sumCubedDiff = 0;
        for (double val : data)
        {
            sumCubedDiff += pow((val - stats.mean) / stats.stddev, 3);
        }
        stats.skewness = sumCubedDiff / data.size();

        // Kurtosis
        double sumQuadDiff = 0;
        for (double val : data)
        {
            sumQuadDiff += pow((val - stats.mean) / stats.stddev, 4);
        }
        stats.kurtosis = (sumQuadDiff / data.size()) - 3;

        return stats;
    }

    // Fungsi untuk membuat histogram
    void createHistogram(const vector<double> &data, int numBins)
    {
        if (data.empty())
            return;

        double minVal = *min_element(data.begin(), data.end());
        double maxVal = *max_element(data.begin(), data.end());
        double binWidth = (maxVal - minVal) / numBins;

        vector<int> histogram(numBins, 0);

        for (double val : data)
        {
            int bin = min((int)((val - minVal) / binWidth), numBins - 1);
            histogram[bin]++;
        }

        cout << "\nHistogram:" << endl;
        cout << "Bin Range\t\tFrequency\tBar" << endl;
        cout << string(60, '-') << endl;

        int maxFreq = *max_element(histogram.begin(), histogram.end());

        for (int i = 0; i < numBins; i++)
        {
            double binStart = minVal + i * binWidth;
            double binEnd = binStart + binWidth;

            cout << fixed << setprecision(2);
            cout << "[" << binStart << " - " << binEnd << ")\t";
            cout << histogram[i] << "\t\t";

            // Draw bar
            int barLength = (histogram[i] * 30) / maxFreq;
            cout << string(barLength, '*') << endl;
        }
    }

    // Fungsi distribusi normal PDF
    double normalPDF(double x, double mean, double stddev)
    {
        double exponent = -0.5 * pow((x - mean) / stddev, 2);
        return (1.0 / (stddev * sqrt(2 * M_PI))) * exp(exponent);
    }

    // Fungsi distribusi normal CDF
    double normalCDF(double x, double mean, double stddev)
    {
        return 0.5 * (1 + erf((x - mean) / (stddev * sqrt(2))));
    }

    // Fungsi untuk menghitung probabilitas (P(X < x), P(X > x), P(a < X < b))
    void calculateProbabilities(double mean, double stddev, const string &attrName)
    {
        cout << "\n=== Probability Calculations for " << attrName << " ===" << endl;
        cout << "Assuming Normal Distribution with mean=" << mean << ", stddev=" << stddev << endl;

        // Example calculations
        double x1 = mean - stddev;
        double x2 = mean + stddev;

        cout << fixed << setprecision(4);
        cout << "P(X < " << x1 << ") = " << normalCDF(x1, mean, stddev) << endl;
        cout << "P(X > " << x2 << ") = " << 1 - normalCDF(x2, mean, stddev) << endl;
        cout << "P(" << x1 << " < X < " << x2 << ") = "
             << normalCDF(x2, mean, stddev) - normalCDF(x1, mean, stddev) << endl;
    }

    // Fungsi untuk test normalitas (Shapiro-Wilk simplified)
    bool isNormallyDistributed(const vector<double> &data, const Statistics &stats)
    {
        // Simplified normality test based on skewness and kurtosis
        // For normal distribution: skewness ≈ 0, kurtosis ≈ 0
        return (abs(stats.skewness) < 0.5 && abs(stats.kurtosis) < 0.5);
    }

    // Main analysis function
    void analyze()
    {
        cout << "=== KDD Dataset Analysis ===" << endl;

        // Perform correlation analysis
        cout << "\n1. CORRELATION ANALYSIS" << endl;
        vector<CorrelationResult> correlations = performCorrelationAnalysis(trainData);

        cout << "Top 10 most correlated attributes with anomaly classification:" << endl;
        cout << setw(30) << "Attribute" << setw(15) << "Correlation" << endl;
        cout << string(45, '-') << endl;

        for (int i = 0; i < min(10, (int)correlations.size()); i++)
        {
            cout << setw(30) << correlations[i].attribute
                 << setw(15) << fixed << setprecision(4) << correlations[i].correlation << endl;
        }

        // Select top 3 attributes
        vector<int> selectedIndices;
        cout << "\nSelected top 3 attributes for analysis:" << endl;
        for (int i = 0; i < 3 && i < correlations.size(); i++)
        {
            cout << i + 1 << ". " << correlations[i].attribute << endl;
            selectedIndices.push_back(correlations[i].index);
        }

        // Analyze each selected attribute
        for (int i = 0; i < selectedIndices.size(); i++)
        {
            int idx = selectedIndices[i];
            string attrName = attributeNames[idx];

            cout << "\n\n=== ANALYSIS FOR ATTRIBUTE: " << attrName << " ===" << endl;

            // Extract feature data
            vector<double> featureData;
            for (const auto &record : trainData)
            {
                if (idx < record.features.size())
                {
                    featureData.push_back(record.features[idx]);
                }
            }

            // Calculate statistics
            cout << "\n2. DESCRIPTIVE STATISTICS" << endl;
            Statistics stats = calculateStatistics(featureData);

            cout << fixed << setprecision(4);
            cout << "Mean: " << stats.mean << endl;
            cout << "Median: " << stats.median << endl;
            cout << "Mode: " << stats.mode << endl;
            cout << "Standard Deviation: " << stats.stddev << endl;
            cout << "Variance: " << stats.variance << endl;
            cout << "Min: " << stats.min << endl;
            cout << "Max: " << stats.max << endl;
            cout << "Skewness: " << stats.skewness << endl;
            cout << "Kurtosis: " << stats.kurtosis << endl;

            // Create histogram
            cout << "\n3. HISTOGRAM (10 bins)" << endl;
            createHistogram(featureData, 10);

            // Distribution analysis
            cout << "\n4. DISTRIBUTION ANALYSIS" << endl;
            if (isNormallyDistributed(featureData, stats))
            {
                cout << "Data appears to follow a NORMAL distribution" << endl;
            }
            else
            {
                cout << "Data does NOT appear to follow a normal distribution" << endl;
                cout << "Skewness indicates: ";
                if (stats.skewness > 0)
                    cout << "Right-skewed (positive skew)" << endl;
                else
                    cout << "Left-skewed (negative skew)" << endl;
            }

            // Probability calculations
            cout << "\n5. PROBABILITY CALCULATIONS" << endl;
            calculateProbabilities(stats.mean, stats.stddev, attrName);
        }
    }

    void loadData(const string &trainFile, const string &testFile)
    {
        ifstream trainCheck(trainFile);
        if (!trainCheck)
        {
            cerr << "Error: Training file not found: " << trainFile << endl;
            exit(EXIT_FAILURE);
        }
        trainCheck.close();

        ifstream testCheck(testFile);
        if (!testCheck)
        {
            cerr << "Error: Test file not found: " << testFile << endl;
            exit(EXIT_FAILURE);
        }
        testCheck.close();

        cout << "Loading training data..." << endl;
        readARFF(trainFile, trainData);
        cout << "Loaded " << trainData.size() << " training records" << endl;

        cout << "Loading test data..." << endl;
        readARFF(testFile, testData);
        cout << "Loaded " << testData.size() << " test records" << endl;
    }
};

int main()
{
    KDDAnalyzer analyzer;

    // Load data files
    analyzer.loadData("KDDTrain+.arff", "KDDTest+.arff");
    // Perform analysis
    analyzer.analyze();

    return 0;
}