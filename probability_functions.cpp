#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

class ProbabilityDistributions
{
public:
    // ===== DISTRIBUSI NORMAL =====

    // Probability Density Function (PDF) untuk distribusi normal
    static double normalPDF(double x, double mean, double stddev)
    {
        double exponent = -0.5 * pow((x - mean) / stddev, 2);
        return (1.0 / (stddev * sqrt(2 * M_PI))) * exp(exponent);
    }

    // Cumulative Distribution Function (CDF) untuk distribusi normal
    static double normalCDF(double x, double mean, double stddev)
    {
        return 0.5 * (1 + erf((x - mean) / (stddev * sqrt(2))));
    }

    // P(a < X < b) untuk distribusi normal
    static double normalProbBetween(double a, double b, double mean, double stddev)
    {
        return normalCDF(b, mean, stddev) - normalCDF(a, mean, stddev);
    }

    // ===== DISTRIBUSI EKSPONENSIAL =====

    // PDF untuk distribusi eksponensial
    static double exponentialPDF(double x, double lambda)
    {
        if (x < 0)
            return 0;
        return lambda * exp(-lambda * x);
    }

    // CDF untuk distribusi eksponensial
    static double exponentialCDF(double x, double lambda)
    {
        if (x < 0)
            return 0;
        return 1 - exp(-lambda * x);
    }

    // ===== DISTRIBUSI BIMODAL (campuran 2 normal) =====

    // PDF untuk distribusi bimodal
    static double bimodalPDF(double x, double mean1, double stddev1, double mean2, double stddev2, double weight1)
    {
        double pdf1 = normalPDF(x, mean1, stddev1);
        double pdf2 = normalPDF(x, mean2, stddev2);
        return weight1 * pdf1 + (1 - weight1) * pdf2;
    }

    // ===== DISTRIBUSI LOG-NORMAL =====

    // PDF untuk distribusi log-normal
    static double lognormalPDF(double x, double meanLog, double stddevLog)
    {
        if (x <= 0)
            return 0;
        double exponent = -pow(log(x) - meanLog, 2) / (2 * pow(stddevLog, 2));
        return (1.0 / (x * stddevLog * sqrt(2 * M_PI))) * exp(exponent);
    }

    // CDF untuk distribusi log-normal
    static double lognormalCDF(double x, double meanLog, double stddevLog)
    {
        if (x <= 0)
            return 0;
        return 0.5 * (1 + erf((log(x) - meanLog) / (stddevLog * sqrt(2))));
    }

    // ===== FUNGSI UTILITAS =====

    // Menghitung parameter distribusi eksponensial dari data
    static double estimateExponentialLambda(const vector<double> &data)
    {
        double sum = 0;
        for (double x : data)
        {
            sum += x;
        }
        return data.size() / sum;
    }

    // Menghitung parameter distribusi log-normal dari data
    static pair<double, double> estimateLogNormalParams(const vector<double> &data)
    {
        double sumLog = 0;
        double sumLogSquared = 0;
        int count = 0;

        for (double x : data)
        {
            if (x > 0)
            {
                double logX = log(x);
                sumLog += logX;
                sumLogSquared += logX * logX;
                count++;
            }
        }

        double meanLog = sumLog / count;
        double variance = (sumLogSquared / count) - (meanLog * meanLog);
        double stddevLog = sqrt(variance);

        return make_pair(meanLog, stddevLog);
    }

    // Visualisasi distribusi (text-based)
    static void plotDistribution(double (*pdf)(double, double, double),
                                 double param1, double param2,
                                 double xMin, double xMax, int numPoints = 50)
    {
        cout << "Distribution Plot:" << endl;
        cout << "X\tPDF\tBar" << endl;
        cout << string(50, '-') << endl;

        double step = (xMax - xMin) / numPoints;
        double maxPDF = 0;
        vector<pair<double, double>> points;

        // Find max PDF for scaling
        for (int i = 0; i <= numPoints; i++)
        {
            double x = xMin + i * step;
            double y = pdf(x, param1, param2);
            points.push_back(make_pair(x, y));
            maxPDF = max(maxPDF, y);
        }

        // Plot
        for (const auto &point : points)
        {
            cout << fixed << setprecision(3);
            cout << point.first << "\t" << point.second << "\t";

            int barLength = (int)((point.second / maxPDF) * 30);
            cout << string(barLength, '*') << endl;
        }
    }
};

// Contoh penggunaan untuk analisis KDD Dataset
void analyzeKDDAttribute(const string &attributeName, const vector<double> &data, const string &distributionType)
{
    cout << "\n=== Analisis Probabilitas untuk " << attributeName << " ===" << endl;
    cout << "Distribusi yang digunakan: " << distributionType << endl
         << endl;

    if (distributionType == "normal")
    {
        // Hitung mean dan stddev
        double sum = 0, sumSq = 0;
        for (double x : data)
        {
            sum += x;
            sumSq += x * x;
        }
        double mean = sum / data.size();
        double variance = (sumSq / data.size()) - (mean * mean);
        double stddev = sqrt(variance);

        cout << "Parameter distribusi normal:" << endl;
        cout << "Mean = " << mean << endl;
        cout << "Standard Deviation = " << stddev << endl
             << endl;

        // Contoh perhitungan probabilitas
        cout << "Contoh perhitungan probabilitas:" << endl;
        cout << "P(X < " << mean << ") = " << ProbabilityDistributions::normalCDF(mean, mean, stddev) << endl;
        cout << "P(X > " << mean + stddev << ") = " << 1 - ProbabilityDistributions::normalCDF(mean + stddev, mean, stddev) << endl;
        cout << "P(" << mean - stddev << " < X < " << mean + stddev << ") = "
             << ProbabilityDistributions::normalProbBetween(mean - stddev, mean + stddev, mean, stddev) << endl;
    }
    else if (distributionType == "exponential")
    {
        double lambda = ProbabilityDistributions::estimateExponentialLambda(data);

        cout << "Parameter distribusi eksponensial:" << endl;
        cout << "Lambda = " << lambda << endl;
        cout << "Mean = " << 1 / lambda << endl
             << endl;

        // Contoh perhitungan probabilitas
        double median = log(2) / lambda;
        cout << "Contoh perhitungan probabilitas:" << endl;
        cout << "P(X < " << median << ") = " << ProbabilityDistributions::exponentialCDF(median, lambda) << endl;
        cout << "P(X > " << 2 * median << ") = " << 1 - ProbabilityDistributions::exponentialCDF(2 * median, lambda) << endl;
    }
    else if (distributionType == "bimodal")
    {
        // Untuk error rates yang bimodal (peaks at 0 and 1)
        cout << "Distribusi bimodal dengan peaks di 0 dan 1" << endl;
        cout << "Menggunakan campuran 2 distribusi normal:" << endl;
        cout << "- Normal 1: mean=0, stddev=0.1" << endl;
        cout << "- Normal 2: mean=1, stddev=0.1" << endl;
        cout << "- Weight: 0.6 untuk normal 1, 0.4 untuk normal 2" << endl
             << endl;

        // Contoh perhitungan
        double x = 0.5;
        double pdf = ProbabilityDistributions::bimodalPDF(x, 0, 0.1, 1, 0.1, 0.6);
        cout << "PDF at x=" << x << " = " << pdf << endl;
    }
}

// Fungsi khusus untuk deteksi anomali berdasarkan probabilitas
class AnomalyDetector
{
public:
    static bool isAnomaly(double value, double mean, double stddev, double threshold = 0.05)
    {
        // Menggunakan distribusi normal
        // Jika probabilitas nilai tersebut < threshold, maka dianggap anomali
        double pLower = ProbabilityDistributions::normalCDF(value, mean, stddev);
        double pUpper = 1 - pLower;
        double pTwoTailed = 2 * min(pLower, pUpper);

        return pTwoTailed < threshold;
    }

    static double anomalyScore(double value, double mean, double stddev)
    {
        // Skor anomali berdasarkan jarak dari mean dalam satuan standard deviation
        return abs((value - mean) / stddev);
    }
};

// Main function untuk demonstrasi
int main()
{
    // Simulasi data untuk masing-masing atribut

    // 1. dst_host_srv_serror_rate (bimodal: peaks at 0 and 1)
    vector<double> serrorRate = {0, 0, 0, 0.1, 0.2, 0, 0, 1, 1, 1, 0.9, 0, 0, 1, 0.05};
    analyzeKDDAttribute("dst_host_srv_serror_rate", serrorRate, "bimodal");

    // 2. dst_bytes (exponential/log-normal)
    vector<double> dstBytes = {100, 200, 150, 5000, 300, 250, 10000, 500, 1000, 2000};
    analyzeKDDAttribute("dst_bytes", dstBytes, "exponential");

    // 3. Demonstrasi anomaly detection
    cout << "\n=== Demonstrasi Deteksi Anomali ===" << endl;
    double mean = 1000, stddev = 500;
    vector<double> testValues = {900, 1100, 500, 3000, 100, 5000};

    cout << "Menggunakan distribusi normal dengan mean=" << mean << ", stddev=" << stddev << endl;
    cout << "Threshold untuk anomali: p < 0.05" << endl
         << endl;

    for (double val : testValues)
    {
        bool anomaly = AnomalyDetector::isAnomaly(val, mean, stddev);
        double score = AnomalyDetector::anomalyScore(val, mean, stddev);

        cout << "Nilai: " << val << " -> ";
        cout << "Anomaly Score: " << fixed << setprecision(2) << score << ", ";
        cout << "Status: " << (anomaly ? "ANOMALY" : "NORMAL") << endl;
    }

    // 4. Plot distribusi
    cout << "\n=== Plot Distribusi Normal ===" << endl;
    ProbabilityDistributions::plotDistribution(ProbabilityDistributions::normalPDF, mean, stddev, 0, 2000);

    return 0;
}