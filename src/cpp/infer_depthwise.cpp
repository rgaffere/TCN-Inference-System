/*
Model Architecture

Input: 6 channels
Layers: 7
Channels per layer: 16
Kernel size: 3
Dilation: 1,2,4,8,16,32,64
Output: 6 channels (prediction)
Mode: streaming (latest point only)

this one is a little different because it only does channel mixing at the inputs and outputs
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <iomanip>
#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

const int inChannels = 6;
const int layerCount = 7;
const int cpl = 16;
const int k = 3;
const int dilations[layerCount] = {1, 2, 4, 8, 16, 32, 64};
const int outChannels = 6;
const int T = 512; // chosen to cover RF = 1 + 2 * (k - 1) * sum(dilations) = 509

double meanVals[inChannels];
double stdVals[inChannels];

double input[inChannels][T];
double midLayers[layerCount][cpl][T];
double hiddenLayers[layerCount][cpl][T];

double inputFilter1[cpl][inChannels][k];
double inputBias1[cpl];

double inputFilter2[cpl][k];
double inputBias2[cpl];

double inputResidualFilter[cpl][inChannels];
double inputResidualBias[cpl];

double hiddenFilter1[layerCount - 1][cpl][k];
double hiddenBias1[layerCount - 1][cpl];

double hiddenFilter2[layerCount - 1][cpl][k];
double hiddenBias2[layerCount - 1][cpl];

double outputFilter[outChannels][cpl];
double outputBiases[outChannels];

int head = T - 1;
int samplesSeen = 0;

void ReLU(double &x)
{
    if (x < 0)
        x = 0;
}

void buildIdx(int idx[k], int d)
{
    for (int i = 0; i < k; i++)
    {
        int delay = d * i;
        idx[i] = (delay < samplesSeen) ? (head - delay + T) & (T - 1) : -1; // bitmask wrap around works since 512 is a power of 2
    }
}

void dilatedConv(
    double *in,
    double *kernel,
    double &acc,
    const int idx[k])
{
    for (int i = 0; i < k; i++)
    {
        if (idx[i] < 0)
            break;
        acc += kernel[k - 1 - i] * in[idx[i]];
    }
}

void doInputLayer()
{
    int latest = head;
    int d = dilations[0];

    int idx[k];
    buildIdx(idx, d);

    // Conv1: mixed 6 -> 16
    for (int oc = 0; oc < cpl; oc++)
    {
        midLayers[0][oc][latest] = inputBias1[oc];

        for (int ic = 0; ic < inChannels; ic++)
            dilatedConv(input[ic], inputFilter1[oc][ic], midLayers[0][oc][latest], idx);

        ReLU(midLayers[0][oc][latest]);
    }

    // Conv2: depthwise 16 -> 16
    for (int ch = 0; ch < cpl; ch++)
    {
        hiddenLayers[0][ch][latest] = inputBias2[ch];

        dilatedConv(midLayers[0][ch], inputFilter2[ch], hiddenLayers[0][ch][latest], idx);

        ReLU(hiddenLayers[0][ch][latest]);

        // Residual projection: mixed 6 -> 16
        double res = inputResidualBias[ch];

        for (int ic = 0; ic < inChannels; ic++)
            res += inputResidualFilter[ch][ic] * input[ic][latest];

        hiddenLayers[0][ch][latest] += res;
    }
}

void doHiddenLayer(int layerNum)
{
    int latest = head;
    int d = dilations[layerNum];
    int f = layerNum - 1;

    int idx[k];
    buildIdx(idx, d);

    // Conv1: depthwise 16 -> 16, no channel mixing
    for (int ch = 0; ch < cpl; ch++)
    {
        midLayers[layerNum][ch][latest] = hiddenBias1[f][ch];

        dilatedConv(hiddenLayers[layerNum - 1][ch], hiddenFilter1[f][ch], midLayers[layerNum][ch][latest], idx);

        ReLU(midLayers[layerNum][ch][latest]);
    }

    // Conv2: depthwise 16 -> 16, no channel mixing
    for (int ch = 0; ch < cpl; ch++)
    {
        hiddenLayers[layerNum][ch][latest] = hiddenBias2[f][ch];

        dilatedConv(midLayers[layerNum][ch], hiddenFilter2[f][ch], hiddenLayers[layerNum][ch][latest], idx);

        ReLU(hiddenLayers[layerNum][ch][latest]);

        // Identity residual: channel ch -> channel ch
        hiddenLayers[layerNum][ch][latest] += hiddenLayers[layerNum - 1][ch][latest];
    }
}

void doOutputLayer(double output[outChannels])
{
    int latest = head;

    for (int oc = 0; oc < outChannels; oc++)
    {
        output[oc] = outputBiases[oc];

        for (int ic = 0; ic < cpl; ic++)
            output[oc] += outputFilter[oc][ic] * hiddenLayers[layerCount - 1][ic][latest];
    }
}

void shiftInput(double newSample[inChannels])
{
    head = (head + 1) & (T - 1);

    if (samplesSeen < T)
        samplesSeen++;

    for (int ch = 0; ch < inChannels; ch++)
        input[ch][head] = newSample[ch];
}

void inferNext(double newSample[inChannels], double prediction[outChannels])
{
    shiftInput(newSample);
    doInputLayer();

    for (int layer = 1; layer < layerCount; layer++)
        doHiddenLayer(layer);

    doOutputLayer(prediction);
}

double computeError(double prediction[outChannels], double actual[outChannels])
{
    double err = 0.0;

    for (int i = 0; i < outChannels; i++)
    {
        double diff = actual[i] - prediction[i];
        err += diff * diff;
    }

    return err;
}

// the inference math stops here, the following is for weight loading and running the inference

void loadWeights(const string& path)
{
    ifstream f(path);
    json j;
    f >> j;

    for (int i = 0; i < inChannels; i++) {
        meanVals[i] = j["mean"][i];
        stdVals[i] = j["std"][i];
    }

    for (int oc = 0; oc < cpl; oc++) {
        inputBias1[oc] = j["inputBias1"][oc];
        inputBias2[oc] = j["inputBias2"][oc];
        inputResidualBias[oc] = j["inputResidualBias"][oc];

        for (int ic = 0; ic < inChannels; ic++) {
            inputResidualFilter[oc][ic] = j["inputResidualFilter"][oc][ic];

            for (int kk = 0; kk < k; kk++)
                inputFilter1[oc][ic][kk] = j["inputFilter1"][oc][ic][kk];
        }

        for (int kk = 0; kk < k; kk++)
            inputFilter2[oc][kk] = j["inputFilter2"][oc][kk];
    }

    for (int l = 0; l < layerCount - 1; l++)
        for (int ch = 0; ch < cpl; ch++) {
            hiddenBias1[l][ch] = j["hiddenBias1"][l][ch];
            hiddenBias2[l][ch] = j["hiddenBias2"][l][ch];

            for (int kk = 0; kk < k; kk++) {
                hiddenFilter1[l][ch][kk] = j["hiddenFilter1"][l][ch][kk];

                hiddenFilter2[l][ch][kk] = j["hiddenFilter2"][l][ch][kk];
            }
        }

    for (int oc = 0; oc < outChannels; oc++) {
        outputBiases[oc] = j["outputBiases"][oc];
        for (int ic = 0; ic < cpl; ic++)
            outputFilter[oc][ic] = j["outputFilter"][oc][ic];
    }
}

int main(int argc, char* argv[])
{
    loadWeights(argv[2]);

    ifstream in(argv[1]);
    ofstream out(argv[3]);

    out << "sample,pred0,pred1,pred2,pred3,pred4,pred5\n";
    out << fixed << setprecision(10);

    string line;
    int sample = 0;

    while (getline(in, line)) {
        stringstream ss(line);
        string cell;
        vector<double> c;

        while (getline(ss, cell, ','))
            c.push_back(stod(cell));

        double x[inChannels] = {
            (c[10] - meanVals[0]) / stdVals[0],
            (c[11] - meanVals[1]) / stdVals[1],
            (c[12] - meanVals[2]) / stdVals[2],
            (c[4]  - meanVals[3]) / stdVals[3],
            (c[5]  - meanVals[4]) / stdVals[4],
            (c[6]  - meanVals[5]) / stdVals[5]
        };

        double pred[outChannels];
        inferNext(x, pred);

        if (sample >= T) {
            out << sample;
            for (int ch = 0; ch < outChannels; ch++)
                out << "," << pred[ch];
            out << "\n";
        }

        sample++;
    }

    return 0;
}