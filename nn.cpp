#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <random>

using namespace std ;

class MLP {
public:
    Eigen::MatrixXd W; // initialisation des poids
    Eigen::VectorXd b; //intialisation des biais 
    Eigen::MatrixXd x;  // matrice d'entrée du batch

MLP(int din, int dout) {

        W = Eigen::MatrixXd::(2 *Random(dout, din) - 1) * (sqrt(6.0) / sqrt(din + dout));
        b = Eigen::VectorXd::(2 *Random(dout)- 1) * (sqrt(6.0) / sqrt(din + dout));

Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        this->x = x;
        return x * W.transpose() + b.transpose(); 

Eigen::MatrixXd backward(const Eigen::MatrixXd& gradout) {
        Eigen::MatrixXd deltaW = gradout.transpose() * x;
        Eigen::VectorXd deltab = gradout.rowwise().sum();
        return gradout * W;
    }
};

// le modèle séquentiel
class SequentialNN {
public:
    vector<MLP> blocks;

    SequentialNN(const vector<MLP>& blocks) : blocks(blocks) {}

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        Eigen::MatrixXd result = x;
        for (const auto& block : blocks) {
            result = block.forward(result);
        }
        return result;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradout) {
        Eigen::MatrixXd result = gradout;
        for (auto i = blocks.rbegin(); i != blocks.rend(); ++i) {
            result = i->backward(result);
        }
        return result;
    }
};

class ReLU {
public:
    Eigen::MatrixXd x;

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        this->x = x;
        return x.array().max(0.0);
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradout) {
        Eigen::MatrixXd new_grad = gradout;
        new_grad.array() *= (x.array() > 0.0).cast<double>();
        return new_grad;
    }
};

class LogSoftmax {
public:
    Eigen::MatrixXd x;

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        this->x = x;
        return x - logsumexp(x.array(), 1).matrix().replicate(1, x.cols());
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradout) {
        Eigen::MatrixXd gradients = Eigen::MatrixXd::Identity(x.cols(), x.cols()) -
                                     (x.array().exp().colwise() / x.array().exp().rowwise().sum()).matrix().replicate(1, x.cols());
        return gradients * gradout;
    }
};

class NLLLoss {
public:
    Eigen::MatrixXd pred;
    std::vector<int> true_labels;

    double forward(const Eigen::MatrixXd& pred, const std::vector<int>& true_labels) {
        this->pred = pred;
        this->true_labels = true_labels;

        double loss = 0.0;
        for (int b = 0; b < pred.rows(); ++b) {
            loss -= pred(b, true_labels[b]);
        }
        return loss;
    }

    Eigen::MatrixXd backward() {
        int din = pred.cols();
        Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(pred.rows(), din);
        for (int b = 0; b < pred.rows(); ++b) {
            jacobian(b, true_labels[b]) = -1.0;
        }
        return jacobian;
    }

    double operator()(const Eigen::MatrixXd& pred, const std::vector<int>& true_labels) {
        return forward(pred, true_labels);
    }
};

class Optimizer {
public:
    double ta;
    SequentialNN compound_nn;

    Optimizer(double ta, const SequentialNN& compound_nn) : ta(ta), compound_nn(compound_nn) {} //Le constructeur prend en paramètre le taux d'apprentissage (ta) et une référence constante à un objet SequentialNN (compound_nn). Il initialise les membres de la classe ta et compound_nn avec les valeurs fournies.

    void step() {
        for (auto& block : compound_nn.blocks) {
            if (typeid(block) == typeid(MLP)) {
                auto& mlp_block = dynamic_cast<MLP&>(block);
                mlp_block.W -= ta * mlp_block.deltaW.transpose();
                mlp_block.b -= ta * mlp_block.deltab.transpose();
            }
        }
    }
};

vector<int> random_batch_indices(int batch_size, int data_size) {
    vector<int> indices(data_size);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    indices.resize(batch_size);
    return indices;
}

 vector<vector<int>> data(60000, vector<int>(784, 0)); //initialisation des 60.000 images d'apprentissage,28*28=784pixels en noir et blanc
    for (int i = 0; i < 60000; ++i) {
        for (int j = 0; j < 784; ++j) {
            data[i][j] = rand() % 256;
        }
    }

    return data;
}

int main() {
    
    MatrixXd trainX = load_data();
    VectorXi trainy(60000); 
    MatrixXd testX = load_data();
    VectorXi testy(10000);
//Normalisation des datas 
    trainX = (trainX.array() - 127.5) / 127.5;
    testX = (testX.array() - 127.5) / 127.5;
     trainX.resize(trainX.rows(), 28 * 28);

        SequentialNN mlp({
            MLP(28 * 28, 128), ReLU(),
            MLP(128, 64), ReLU(),
            MLP(64, 10), LogSoftmax()
        });

        Optimizer optimizer(1e-3, mlp);

        NLLLoss loss_function;
        for (int epoch = 0; epoch < 14000; ++epoch) {
            auto batch_indices = random_batch_indices(100, data.size());
            Eigen::MatrixXd x_batch(100, 784);
            vector<int> y_batch(100);
            for (int i = 0; i < 100; ++i) {
                x_batch.row(i) = Eigen::Map<Eigen::VectorXd>(data[batch_indices[i]].data(), 784);
                y_batch[i] = rand() % 10;
            }

            Eigen::MatrixXd prediction = mlp.forward(x_batch);
            double loss_value = loss_function.forward(prediction, y_batch);
            loss_function.backward();
            mlp.backward(loss_function.backward());
            optimizer.step();
        }

        // Calculez l'exactitude sur le reste du code...
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() <<endl;
    } catch (...) {
        cerr << "Unknown exception caught." << endl;
    }

    return 0;
}





