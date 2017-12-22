import numpy
import random
import math
import collections
import matplotlib.pyplot as plt


#Define input x_i and parameter v_i
n = 300;
x = numpy.random.uniform(0,1,n);
v = numpy.random.uniform(-0.1,0.1,n);

#Define output parameter 'd':
d = [];
for i in range(0,n):
    temp = numpy.sin(20*x[i]) + 3*x[i] + v[i];
    d.append(temp);


#Define Tan Activation Function:
def tan_activation_function(v):
    return numpy.tanh(v);


#Define Activation Function:
def activation_function(v):
    return v;

def feedForwardNetwork(x_input,d_output,weight1,weight2,b1,b2):
    layer2_input = []
    layer2_output = [];
    for i in range(0,len(weight1)):
        temp = tan_activation_function((x_input * weight1[i]) + b1[i]);
        layer2_input.append((x_input * weight1[i]) + b1[i]);
        layer2_output.append(temp);
        
    v_vector = numpy.array(layer2_output);
    v_vector.resize(1,24);
    weight2.resize(24,1);
    
    y = activation_function(numpy.dot(v_vector,weight2) + b2);
    return layer2_input,layer2_output,y;


learning_rate = 0.01;
def feedBackNetwork(x_input,d_output,layer2_input,layer2_output,y_output,weight1,weight2,b1,b2):
    for i in range(0,len(weight2_initial)):
        #Weight 2 update:
        w2_update = -2 * learning_rate * ((d_output - y_output) * layer2_output[i]);
        weight2[i] = weight2[i] - w2_update;

        #Weight 1 update:
        w1_update = (1 - math.pow(numpy.tanh(layer2_input[i]),2));
        weight1[i] = weight1[i] - (-2 * learning_rate * weight2[i] * (d_output - y_output) * w1_update * x_input);
        
        #Update Bias 1:
        bias_1[i] = bias_1[i] - (-2 * learning_rate * weight2[i] * (d_output - y_output) * w1_update);
    #Update Bias 2:
    b2 = b2 - (-2 * learning_rate * (d_output - y_output));
    return weight1,weight2,b1,b2;


#Define Random Weights:
weight1_initial = numpy.random.uniform(-10,10,24);
bias_1 = numpy.random.uniform(-7,4,24);
weight2_initial = numpy.random.uniform(-7,8,24);
bias_25 = numpy.random.uniform(-1,1,1)[0];


#Code Snippet to run EPOCHS - online learning - feedforward and feedback networks
epochs = [];
rmse_list = [];
for epoch in range(0,1000):
    error = [];
    total_error = 0;
    rmse = 100;
    for i in range(0,len(x)):
        layer2_input,layer2_output,y = feedForwardNetwork(x[i],d[i],weight1_initial,weight2_initial,bias_1,bias_25);
        error.extend(d[i] - y[0]);
        weight1_initial,weight2_initial,bias_1,bias_25 = feedBackNetwork(x[i],d[i],layer2_input,layer2_output,y,weight1_initial,weight2_initial,bias_1,bias_25);
        temp_w2 = [];
        for element in weight2_initial:
            temp_w2.append(element[0]);
            weight2_initial = numpy.array(temp_w2);

    for k in range(0,len(error)):
        total_error = total_error + (error[k]*error[k]);
        
    rmse = total_error/n;
    
    epochs.append(epoch);
    rmse_list.append(rmse);
    print ("EPOCH:",epoch,"  RMSE:", rmse);
    if rmse <= 0.01:
        break;


#Calculate Predicted Output Parameter:
new_d = [];
for i in range(0,len(x)):
    layer2_input,layer2_output,y = feedForwardNetwork(x[i],d[i],weight1_initial,weight2_initial,bias_1,bias_25);
    new_d.append(y);


#Code to generate original curve
fig, ax = plt.subplots(figsize=(7,7));
plt.scatter(x,d, c = 'orange');
plt.title('Fig 1: Xi input curve');
plt.ylabel('D-->');
plt.xlabel('X-->');
plt.show();

#Code to generate original curve along with fitted(predicted) curve
fig, ax = plt.subplots(figsize=(7,7));
plt.scatter(x,d, c = 'orange');
plt.scatter(x,new_d, c = 'green');
plt.title('Fig 2: X input(Orange) vs NN output(Green)');
plt.ylabel('D-->');
plt.xlabel('X-->');
plt.show();

#Code to Epochs vs RMSE convergence chart
fig, ax = plt.subplots(figsize=(7,7));
ax.plot(epochs,rmse_list, c = 'red');
plt.title('Fig 3: Epochs vs RMSE');
plt.ylabel('RMSE');
plt.xlabel('Number of Epochs');
plt.show();


# Following are the steps taken to implement the algorithm:
# 1. We begin by initializing input 300 values of X randomly chosen between [0,1]
# 
# 2. Next, we define a variable V chosen at random between [-1/10,1/10]. The size is 300.
# 
# 3. Next, we define our output of the NN for the corresponding X as:
# 		di = sin(20xi) + 3xi + νi, i = 1,2,3,.....,300
# 
# 4. We plot the curve shown in Fig 1.
# 
# 5. Next, we define 2 functions. Each correspond to an activation function used by different layers of NN. In this case, the hidden layer uses "tanh" activation function and the output layer uses an activation function defined according to the equation: "u(v) = v"
# 
# 6. Next, we initialize the weights chosen randomly between certain interval. We chose the interval [-10,10]. We have defined 4 sets of weights. First is of size 24x1 and serve as the input weights. Second weight serves as the bias for the hidden layer and is of size 24x1. Third weight serves as the output weight of hidden layer and is of size 24x1. Fourth weight is chosen randomly between the interval [-1,1] and is of size 1x1.
# 
# 7. Next, we define a function with the given arguments "feedForwardNetwork(xi,di,input_layer_wt,hidden_layer_wt,bias1,bias2)". Here xi and di represents 1 input and corresponding desired output.
#     The function feedForwardNetwork calculates the NN output given the input xi. Say that the output is given by 'y'.
#     The code for feefForwardNetwork is:
#     The feedForwardNetwork returns layer2_input(input vector to hidden layer), layer2_output (output vector to hidden layer) and predicted output ‘y’.
#     
# 8. Next, we define a function feedBackNetwork(xi,di,layer2_input,layer2_output,y_output,weight1,weight2,b1,b2):
# 		• xi -> input x.
# 		• di -> desired output
# 		• layer2_input -> input vector to hidden layer
# 		• layer2_output -> output vector to hidden layer
# 		• y_output -> predicted output
# 		• weight1 -> input layer weights
# 		• weight2 -> hidden layer output weights
# 		• b1 -> hidden layer bias
# 		• b2 -> output layer bias
# 
# 9. Next we iterate over all the values of X in a loop and make the following sequential calls:
# 		• feedForwardNetwork (args…);
# 		• feedBackNetwork (args…);
# 
# 	The feedForwardNetwork returns certain parameters which are used as input to the feedBackNetwork. See (7, 8) for the details of parameters.
# 	We also calculate the RMSE when the loop runs over all the values of x as:
# 			(1/n) * Sum from i to n [(di – f(xi,w))^2] 
# 			where di -> Desired Oputput
# 			f(xi, w) -> represents the predicted output y of NN.
# 	We call this entire run as EPOCH 0.
# 
# 10. Repeat 9 for certain number of steps (EPOCHS) until the value of RMSE become less than or equal to 0.01.
