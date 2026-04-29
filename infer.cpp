#include <iostream>
using namespace std;

struct TCNModel
{
    double *hiddenLayers;
    double *filters;
    double *biases;
    int *d;
    int inChannels;
    int outChannels;
    int k;
    int T;
    int numBlocks;
};

void ReLU(
    double &x // Since we're focused on inference, we only do ReLU on the latest point
)
{
    if (x < 0)
        x = 0;
}

void dilatedConv(
    double *in,     // in - input sequence, format : last entry is the latest entry
    double *kernel, // kernel - filter, format : same as the input, last weight is for latest entry
    double &acc,    // acc - accumulator, pass by reference so we dont have an output. cleaner solution
    int T,          // T - input sequence length
    int k,          // k - filter length
    int d           // d - dilation factor
)
{
    for (int i = 0; i < k; i++)
    {
        int index = T - (1 + d * i);
        if (index < 0)
            break;                            // zero padding for when we go beyond the size of the sequnce
        acc += kernel[k - 1 - i] * in[index]; // this skips every point according to the dilation factor
    }
}

void doResiBlock(
    double *in,  // in - input sequence, could be the output of a hidden layer, or the very first layer. depending on channel count is either 1d or 2d
    double *h1,  // h1 - result of first convolution, can be either 1d or 2d depending on channel count
    double *b1,  // b1 - this is a 1d array of biases to be added to in for the computation of h1
    double *k1,  // k1 - this is the filter to be applied to in for the computation of h1
    double *out, // out - this is the result of the second convolution, can be 1d or 2d
    double *b2,  // b2 - this is a 1d array of biases to be applied to h1 in order to compute out
    double *k2,  // k2 - this is the filter to be applied onto h1 in order to compute out
    double *k3,  // k3,b3 - this is for the residual addition.
    double *b3,
    int k,          // k - this is the filter length
    int d,          // d - this is the dilation factor
    int T,          // T - this is the length of the input sequence and h1 and out
    int inChannels, // inChannels - this is the number of input channels (applies to in)
    int outChannels // outChannels - this is the number of output channels (applies to h1 and out)
)
{
    // first convolution
    int latest = T - 1;
    for (int i = 0; i < outChannels; i++)
    {
        h1[T * i + latest] = b1[i];

        for (int j = 0; j < inChannels; j++)
        {
            dilatedConv(&in[j * T], &k1[(i * inChannels + j) * k], h1[T * i + latest], T, k, d);
        }
        ReLU(h1[T * i + latest]);
    }
    // second convolution
    for (int i = 0; i < outChannels; i++)
    {
        out[T * i + latest] = b2[i];

        for (int j = 0; j < outChannels; j++)
        {
            dilatedConv(&h1[j * T], &k2[(i * outChannels + j) * k], out[T * i + latest], T, k, d);
        }
        ReLU(out[T * i + latest]);
    }
    // now do the residual add, if in and out channels are the same, k3 is an identity matrix
    for (int oc = 0; oc < outChannels; oc++)
    {
        double res = b3[oc];

        for (int ic = 0; ic < inChannels; ic++)
        {
            res += k3[oc * inChannels + ic] * in[ic * T + latest];
        }

        out[oc * T + latest] += res;
    }
}

double inferNext // takes relevant data and predicts the next value of the sequence
    (
        double *in, // input here comes from the cache and is already sliced to only include relevant history base on dilation
        TCNModel &model)
{
    doResiBlock(in, &model.hiddenLayers[0], &model.biases[0], &model.filters[0],
                &model.hiddenLayers[model.outChannels * model.T],
                &model.biases[model.outChannels],
                &model.filters[model.outChannels * model.inChannels * model.k],
                &model.filters[(model.outChannels * model.inChannels * model.k) +
                               (model.outChannels * model.outChannels * model.k)],
                &model.biases[2 * model.outChannels],
                model.k, model.d[0], model.T, model.inChannels, model.outChannels);

    int block0Size =
        (model.outChannels * model.inChannels * model.k) +
        (model.outChannels * model.outChannels * model.k) +
        (model.outChannels * model.inChannels);

    int hiddenBlockSize =
        (model.outChannels * model.outChannels * model.k) +
        (model.outChannels * model.outChannels * model.k) +
        (model.outChannels * model.outChannels);

    for (int i = 1; i < model.numBlocks; i++)
    {
        doResiBlock(
            &model.hiddenLayers[(i - 1) * 2 * model.outChannels * model.T + model.outChannels * model.T],
            &model.hiddenLayers[i * 2 * model.outChannels * model.T],
            &model.biases[i * 3 * model.outChannels],
            &model.filters[block0Size + (i - 1) * hiddenBlockSize],

            &model.hiddenLayers[i * 2 * model.outChannels * model.T + model.outChannels * model.T],
            &model.biases[i * 3 * model.outChannels + model.outChannels],
            &model.filters[block0Size + (i - 1) * hiddenBlockSize +
                           (model.outChannels * model.outChannels * model.k)],

            &model.filters[block0Size + (i - 1) * hiddenBlockSize +
                           2 * (model.outChannels * model.outChannels * model.k)],
            &model.biases[i * 3 * model.outChannels + 2 * model.outChannels],

            model.k,
            model.d[i],
            model.T,
            model.outChannels,
            model.outChannels);
    }
    return model.hiddenLayers[model.outChannels * model.numBlocks + model.T - 1];
}
