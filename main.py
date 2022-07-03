from layers import ActivationLayer, FCLayer, Network, mse, mse_prime, tanh_prime, tanh
from functions import get_data

import pickle

train_data = get_data()
test_data = get_data(type='test')
print("Train and test data gotten successfully, saving to files")

pickle.dump( train_data, open(r"E:\personal_projects\breast cancer detection\code\pickles\train_data.p", 'wb') )
pickle.dump( test_data, open(r"E:\personal_projects\breast cancer detection\code\pickles\test_data.p", 'wb') )

X_train = train_data["Image"]
y_train = train_data["Label"]

X_test = test_data["Image"]
y_test = test_data["Label"]

net = Network()
net.add(FCLayer( 640*480 + 1, 500 )) # input_shape=(1, 640*480); output_shape=(1, 500)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(500, 250))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(250, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(10, 2))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(X_train, y_train, epochs=35, learning_rate=0.1)

output = net.predict(X_test)

print("Pickling important items")
pickle.dump( net, open(r"E:\personal_projects\breast cancer detection\code\pickles\neural_net.p", 'wb') )
pickle.dump( y_test, open(r"E:\personal_projects\breast cancer detection\code\pickles\test_results.p", 'wb') )
pickle.dump( output, open(r"E:\personal_projects\breast cancer detection\code\pickles\predicted_test_results.p", 'wb') )