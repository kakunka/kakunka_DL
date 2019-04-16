// backpropagation alogrithm for mlp
// author: kakunka
#include <stdio.h>
#include<stdlib.h>
#include<iostream>
#include<assert.h>
#include<random>
#include<cmath>
#include<vector>
// #include"mnist.cpp"


using namespace std;

template <typename T>
inline int get_array_len(T &p)
{
    return sizeof(T) / sizeof(*p);
}

inline float generateRandomFloat(){
    return (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) ;
}

class neuron{
    float * weights;
    float bias;
    int weightSize;
public:

    void setNeuron(int n){
        cout<<"init neuron  "<<n<<endl;
        weights = new float[n];
        weightSize = n;
    }
    ~neuron(){
        delete []weights;
    }
};
class layer{
    neuron * neuronsOfLayer;
    int size;
    float * values;
public:
    void setLayer(int n, int connections){
        size = n;
        neuronsOfLayer = new neuron[n];
        values = new float[n];
        for (int i=0; i<n; i++){
            neuronsOfLayer[i].setNeuron(connections);
        }
    }

};


class MLP{
    layer * hiddebLayers;
    layer inputLayer;
    layer outputLayer;
    int size_layer;
public:
    MLP(int mlp_struct_[], int n){
        assert(n>=3);
        outputLayer.setLayer(mlp_struct_[n-1], 0);
        hiddebLayers = new layer[n-2];
        for (int i=n-2; i>0; i--){
             hiddebLayers[i].setLayer(mlp_struct_[i], mlp_struct_[i+1]);
        }
        inputLayer.setLayer(mlp_struct_[0], mlp_struct_[1]);
        size_layer = n;
    }

    void forwardPropagation(int x[], int n){

        for (int i=0; i<size_layer; i++){
            
        }
    }

};

class MLP_Matrix{
    float *** weightMatrix;
    float *** deltaWeightMatrix;
    float ** neuronValues;
    float ** deltaZ; 
    int size;
    int * mlp_struct;
    float alpha = 0.6;
    float eta = 0.2;
public: 
    MLP_Matrix(int mlp_struct_[], int n){
        assert(n>=3);
        size = n;
        mlp_struct = mlp_struct_;

        weightMatrix = new float**[n-1];
        deltaWeightMatrix = new float**[n-1];
        for (int i=0; i<n-1; i++){
            weightMatrix[i] = new float*[mlp_struct_[i]];
            deltaWeightMatrix[i] = new float*[mlp_struct_[i]];
            for(int j = 0; j < mlp_struct_[i]; j++)
            {
                weightMatrix[i][j] = new float[mlp_struct_[i+1]];
                deltaWeightMatrix[i][j] = new float[mlp_struct_[i+1]];
                for (int n=0; n<mlp_struct_[i+1]; n++){
                    weightMatrix[i][j][n] = generateRandomFloat(); //init using random float value (-1,1)
                }
            }
            
        }
        
        neuronValues = new float*[n];
        deltaZ = new float*[n];

        for (int i=0; i<n; i++){    
            neuronValues[i] = new float[mlp_struct_[i]];
            deltaZ[i] = new float[mlp_struct_[i]];
        }


    }

    void forwardPropagation(float inputX[], int inputSize){
        assert(*mlp_struct == inputSize);
        for (int i=0; i<inputSize; i++){
            neuronValues[0][i] = inputX[i];
            assert(!isnan(neuronValues[0][i]));
        }
        // neuronValues[0] = inputX; //cause pointer being freed was not allocated when delete
        for (int i=1; i<size; i++){
            for (int j=0; j<*(mlp_struct+i); j++){
                neuronValues[i][j] = 0;
                for(int n = 0; n < *(mlp_struct+i-1); n++)
                {
                    neuronValues[i][j] += neuronValues[i-1][n] * weightMatrix[i-1][n][j];
                    if (isnan(neuronValues[i][j])){
                        cout<<" i:"<<i<<" j:"<<j<<" n:"<<n<<endl;
                        cout<<"neuronValues[i-1][n]:"<<neuronValues[i-1][n] <<" weightMatrix[i-1][n][j]:"<<weightMatrix[i-1][n][j]<<endl;
                    }
                    assert(!isnan(neuronValues[i][j]));
                }
                
                neuronValues[i][j] = sigmoid(neuronValues[i][j]);
                
            }
        }
    }

    void backpropagation(float yData[], int outputSize){
        assert(*(mlp_struct+size-1) == outputSize);
        // two plan:    1.update weight when computing deltaZ
        //              2.update after  all deltaZ ready  
        for (int i=0; i<mlp_struct[size-1]; i++){
            // output layer
            deltaZ[size-1][i] = partialDerivativeCostFunction(neuronValues[size-1][i], yData[i]) * partialDerivativeActivationFunction(neuronValues[size-1][i]);
            assert(!isinf(deltaZ[size-1][i]));
            // 1. update right away    
            // for (int n=0; n<mlp_struct[size-2]; n++){
            //     weightMatrix[size-2][n][i] += neuronValues[size-2][n] * deltaZ[size-1][i];
            // }
        }
        for (int i=size-2; i>0; i--){
            for (int j=0; j<mlp_struct[i]; j++){
                float sum = 0;
                for (int n=0; n<mlp_struct[i+1]; n++){
                    sum += deltaZ[i+1][n] * weightMatrix[i][j][n];
                }
                deltaZ[i][j] = sum * partialDerivativeActivationFunction(neuronValues[i][j]);
                if (isinf(deltaZ[i][j])){
                    cout<<"stop";
                }
                assert(!isinf(deltaZ[i][j]));
                // 1. update right away  
                // for (int n=0; n<mlp_struct[i-1]; n++){
                //     weightMatrix[i-1][n][j] += neuronValues[i][n] * deltaZ[i+1][j];
                // }
            }
        }

        // 2. update after  all deltaZ ready  
        for (int i=0; i<size-1; i++){
            for(int j=0; j<mlp_struct[i]; j++){
                for (int n=0; n<mlp_struct[i+1]; n++){
                    float gradient = neuronValues[i][j] * deltaZ[i+1][n];
                    float olddeltaWeight = deltaWeightMatrix[i][j][n];
                    float newdeltaWeight = alpha*olddeltaWeight* + eta*neuronValues[i][j]*gradient;
                    deltaWeightMatrix[i][j][n] = newdeltaWeight;
                    weightMatrix[i][j][n] += newdeltaWeight;
                    if (isnan(weightMatrix[i][j][n])){
                        cout<<" i:"<<i<<" j:"<<j<<" n:"<<n<<endl;
                        cout<<" neuronValues[i][j]:"<<neuronValues[i][j] << " deltaZ[i+1][n]:"<<deltaZ[i+1][n]<<endl;
                    }
                    assert(!isnan(weightMatrix[i][j][n]));
                }
            }
        }
    }

    float partialDerivativeCostFunction(float yData, float yOutput){
        // compute the partial derivative of cost function wrt y
        // using mean square error as cost function
        return (yOutput - yData) ;// (* 2 / mlp_struct[size-1])  ;
    }
    float partialDerivativeActivationFunction(float z){
        // using sigmoid function  as activation function
        return sigmoid(z) *(1-sigmoid(z));
    }
   

    float costMSE(float y[],int n){
        float c = 0;
        // cout<<"cost:  \t";
        for (int i=0; i<n; i++){
            // cout<<"neuronValues[size-1][i]:"<<neuronValues[size-1][i]<<" y[i]:"<<y[i]<<"\t";
            c +=pow(neuronValues[size-1][i] - y[i], 2);
        }
        c /= float(n);
        // cout<<endl;
        return c;
    }

    float sigmoid(float x){
        // assert(!isnan(1 / (1+exp(-x))));
        return 1 / (1+exp(-x));
    }

    int predictSoftmax(){
        float max = 0;
        int maxIndex = 0;
        for (int i=0; i<mlp_struct[size-1]; i++){
            if (max < neuronValues[size-1][i]){
                maxIndex = i;
                max = neuronValues[size-1][i];
            }
        }
        return maxIndex;
    }

    void  output(){
        // float * data, int& n
        // data = neuronValues[size-1];
        // n = mlp_struct[size-1];

        for(int i=0; i< mlp_struct[size-1]; i++){
            cout<<*(neuronValues[size-1]+i)<<" ";
        }
        cout<<endl;
    }

    void String(){
        cout<<"struct "<<endl;
        for (int i=0; i<size; i++){
            cout<<*(mlp_struct+i)<<" ";
        }
        cout<<endl;
        cout<<"weight: "<<endl;
        for (int i=0; i<size-1; i++){
            cout<<"layer "<<i<<" ";
            for (int j=0; j<*(mlp_struct+i); j++){
                cout<<"neuron"<<j<<" ";
                for (int n=0; n<*(mlp_struct+i+1); n++){
                    cout<<weightMatrix[i][j][n]<<" ";
                }
            }  
            cout<<endl;
        } 
        cout<<"values: "<<endl;
        for (int i=0; i<size; i++){
            for (int j=0; j<*(mlp_struct+i); j++){
                cout<<neuronValues[i][j]<<" ";
            }
            cout<<endl;
        } 
    }

    ~MLP_Matrix(){
        cout<<"release resource ..... ";
        for (int i=0; i<size-1; i++){
            for(int j = 0; j < mlp_struct[i]; j++)
            {
                delete[] weightMatrix[i][j];
                delete[] deltaWeightMatrix[i][j];

            }
            delete[] weightMatrix[i];
            delete[] deltaWeightMatrix[i];
        }
        delete weightMatrix;
        delete deltaWeightMatrix;

        for (int i=0; i<size; i++){   
            delete[] neuronValues[i];
        }
        delete[] neuronValues;
    }
};

void test(vector<float>testlabels, vector<vector <float> >testimages, MLP_Matrix &m){
    int const outputSize = 10;
    int const inputSize = 784;
    int testEpochs = 1000;
    int errorNumber = 0;
    int testDataSize = testlabels.size();
    int index = 0;
    float y[outputSize] = {};
    float x[inputSize] = {};
     for (int i=0; i<testEpochs; i++){
        index = rand()%testDataSize;
        for (int j=0; j<testimages[index].size(); j++){
            x[j] = testimages[index][j];
        }
        m.forwardPropagation(x, inputSize);
        if (testlabels[index] != m.predictSoftmax() ){
            errorNumber++;
        }
    }
    cout<<"错误比例\t"<<float(errorNumber)/float(testEpochs)<<"\n错误个数\t"<<errorNumber<<endl;
}

int main(){

    int a[11] = {2, 100, 100, 1024, 1024, 100, 100, 100, 10, 10, 1};
    MLP_Matrix m(a, get_array_len(a));
    // m.String();
    
    
    // // 模拟圆 测试
    int const size = 100000;
    int testNumber = 1000;
    float y[size][1] = {};
    float x[size][2] = {};
    for (int i =0; i< size; i++){
        x[i][0] = 2*generateRandomFloat()-1;
        x[i][1] = 2*generateRandomFloat()-1;
        // y[i][0] = pow(x[i][0], 2) + pow(x[i][1], 2);
        y[i][0] = x[i][0] + x[i][1];
    }

    // train
    int index = 0;
    int trainEpochs = size-testNumber;    
    float input[2] = {};
    
    for(int i=0; i<trainEpochs; i++){
        index = rand()%size;
        m.forwardPropagation(x[index], 2);
        m.backpropagation(y[index], 1);
        cout <<"第 "<<i<<" 轮的损失 "<< m.costMSE(y[index], 1)<<endl;
        if (i%5000 == 0 && i>0){
            float cost = 0;
            for (int j=trainEpochs; j<size-800; j++){
                index = j;
                m.forwardPropagation(x[index], 2);
                cost +=  m.costMSE(y[index], 1);
                cout<<" -----------------  "<<cost<<" ----------------   "<<endl;
            }
        }
    }

    // testc
    cout<<"test...."<<endl;
    float cost = 0;
    for (int i=trainEpochs; i<size; i++){
        index = i;
        m.forwardPropagation(x[index], 2);
        cout<<y[index][0]<<endl;
        m.output();
        cost +=  m.costMSE(y[index], 1);
        cout<<" ----------------- "<<endl;
    }
    index = rand()%size;
    m.forwardPropagation(x[index], 2);
    cout<<y[index][0]<<endl;
    m.output();
    


//  hand-write digital number
    // vector<float>testlabels;
    // vector<float>labels;

	// read_Mnist_Label("/Users/ligang/Coding/Language/c++/ML/SimpleNeuralNetwork/data/t10k-labels-idx1-ubyte", testlabels);
	// read_Mnist_Label("/Users/ligang/Coding/Language/c++/ML/SimpleNeuralNetwork/data/train-labels-idx1-ubyte", labels);



	// vector<vector <float> >testimages;
    // vector<vector <float> >images;
	// read_Mnist_Images("/Users/ligang/Coding/Language/c++/ML/SimpleNeuralNetwork/data/t10k-images-idx3-ubyte", testimages);
    // read_Mnist_Images("/Users/ligang/Coding/Language/c++/ML/SimpleNeuralNetwork/data/train-images-idx3-ubyte", images);

    // int const outputSize = 10;
    // int const inputSize = 784;
    // float y[outputSize] = {};

    // float x[inputSize] = {0};
    // int index = 0;

    // //  train
    // int dataSize = labels.size();
    // int epochs = 50001;
    // for (int i=0; i<images.size(); i++){
    //     index = i;
    //     // index = i;
    //     // one-hot encoding
    //     for (int j=0; j<outputSize; j++){
    //         if (float(j)==labels[index]){
    //             y[j] = 1.0;
    //         }else{
    //             y[j] = 0.0;
    //         }
    //     }
    //     for (int j=0; j<inputSize; j++){
    //         x[j] = images[index][j];
    //     }
    //     m.forwardPropagation(x, inputSize);
    //     m.backpropagation(y, outputSize);
    //     cout <<"第 "<<i<<" 轮的损失 "<< m.costMSE(y, outputSize)<<endl;
    //     if (i % 5000 == 0 && i>0){
    //         test(testlabels, testimages, m);
    //     }
    // }


    // // test
    // // int testEpochs = 1000;
    // // int errorNumber = 0;
    // int testDataSize = testlabels.size();
    // //  for (int i=0; i<testEpochs; i++){
    // //     index = rand()%testDataSize;
    // //     for (int j=0; j<testimages[index].size(); j++){
    // //         x[j] = testimages[index][j];
    // //     }
    // //     m.forwardPropagation(x, inputSize);
    // //     if (testlabels[index] != m.predictSoftmax() ){
    // //         errorNumber++;
    // //     }
    // // }
    // // cout<<"错误比例\t"<<float(errorNumber)/float(testEpochs)<<"\n错误个数\t"<<errorNumber<<endl;

    // index = rand()%testDataSize;
    // for (int j=0; j<testimages[index].size(); j++){
    //     x[j] = testimages[index][j];
    // }
    // m.forwardPropagation(x, inputSize);
    // cout<<"testlabels[index] "<<testlabels[index]<<endl;
    // cout<<m.predictSoftmax()<<endl;
    // m.output();
    // float *data;
    // int size = 0;
    // m.output(data, size);
    // cout<<size;
    // for(int i=0; i<size; i++){
    //     cout<<*(data+i)<<" "<<endl;
    // }
    

    // test(testlabels, testimages, m);
    
    return 0;
}