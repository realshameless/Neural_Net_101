import xlrd
import xlwt
import numpy as np


workbook = xlrd.open_workbook("F:\\1.xls")
sheet = workbook.sheet_by_name("1")
data = []           # height, weight, index
all_y_trues = []    # gender
test_data = []      # height, weight, index
test_true = []      # gender
i = 1
while i<251:
    data.append([sheet.cell(i,1).value,sheet.cell(i,1).value,sheet.cell(i,1).value])
    all_y_trues.append(sheet.cell(i,0).value)
    i += 1
i = 253
while i< 503:
    test_data.append([sheet.cell(i,1).value,sheet.cell(i,1).value,sheet.cell(i,1).value])
    test_true.append(sheet.cell(i,0).value)
    i += 1

workbook.release_resources()
data = np.array(data)
test_data = np.array(test_data)
all_y_trues = np.array(all_y_trues)
test_true = np.array(test_true)


def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1-fx)

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNet:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1)
        h2 = sigmoid(self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2)
        h3 = sigmoid(self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3)
        o1 = sigmoid(self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4)
        return o1

    def train(self,data,all_y_trues):

        learn_rate = 0.1
        epochs = 100000
        for epoch in range(epochs):
            for x,y_true in zip(data,all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.w3 * x[2] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w4 * x[0] + self.w5 * x[1] + self.w6 * x[2] + self.b2
                h2 = sigmoid(sum_h2)

                sum_h3 = self.w7 * x[0] + self.w8 * x[1] + self.w9 * x[2] + self.b3
                h3 = sigmoid(sum_h3)

                sum_o1 = self.w10 * h1 + self.w11 * h2 + self.w12 * h3 + self.b4
                o1 = sigmoid(sum_o1)
                y_pred = o1


                dL_dy_pred = -2 * (y_true - y_pred)

                # for the neuron o1
                dy_pred_dw10 = h1 * deriv_sigmoid(sum_o1)
                dy_pred_dw11 = h2 * deriv_sigmoid(sum_o1)
                dy_pred_dw12 = h3 * deriv_sigmoid(sum_o1)
                
                dypred_db4 = deriv_sigmoid(sum_o1)

                # for the neuron h1
                dy_pred_dh1 = self.w10 * deriv_sigmoid(sum_o1)

                dh1_dw1 = x[0] * deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1] * deriv_sigmoid(sum_h1)
                dh1_dw3 = x[2] * deriv_sigmoid(sum_h1)
                
                dh1_db1 = deriv_sigmoid(sum_h1)
                
                # for the neuron h2
                dy_pred_dh2 = self.w11 * deriv_sigmoid(sum_o1)

                dh2_dw4 = x[0] * deriv_sigmoid(sum_h2)
                dh2_dw5 = x[1] * deriv_sigmoid(sum_h2)
                dh2_dw6 = x[2] * deriv_sigmoid(sum_h2)
                
                dh2_db2 = deriv_sigmoid(sum_h2)

                # for the neuron h3
                dy_pred_dh3 = self.w12 * deriv_sigmoid(sum_o1)

                dh3_dw7 = x[0] * deriv_sigmoid(sum_h3)
                dh3_dw8 = x[1] * deriv_sigmoid(sum_h3)
                dh3_dw9 = x[2] * deriv_sigmoid(sum_h3)
                
                dh3_db3 = deriv_sigmoid(sum_h3)

                #update the weights and biases
                self.w1 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_dw1
                self.w2 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_dw2
                self.w3 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_dw3
                self.w4 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_dw4
                self.w5 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_dw5
                self.w6 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_dw6
                self.w7 -= learn_rate * dL_dy_pred * dy_pred_dh3 * dh3_dw7
                self.w8 -= learn_rate * dL_dy_pred * dy_pred_dh3 * dh3_dw8
                self.w9 -= learn_rate * dL_dy_pred * dy_pred_dh3 * dh3_dw9

                self.w10 -= learn_rate * dL_dy_pred * dy_pred_dw10
                self.w11 -= learn_rate * dL_dy_pred * dy_pred_dw11
                self.w12 -= learn_rate * dL_dy_pred * dy_pred_dw12

                self.b1 -= learn_rate * dL_dy_pred * dy_pred_dh1 * dh1_db1
                self.b2 -= learn_rate * dL_dy_pred * dy_pred_dh2 * dh2_db2
                self.b3 -= learn_rate * dL_dy_pred * dy_pred_dh3 * dh3_db3
                self.b4 -= learn_rate * dL_dy_pred * dypred_db4

            if epoch %1000 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues,y_preds)
                print("epoch %d loss: %f \n" %(epoch,loss))

network = NeuralNet()
network.train(data,all_y_trues)


for i in range(250):
    test_loss_minus = ((network.feedforward(test_data[i]) - test_true[i]) **2).mean()
    print("test No. %d loss to the real number %f"%(i,test_loss_minus))

         

                


                

            
            
    

        
