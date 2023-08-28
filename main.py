import torch
import numpy as np
import pickle

from models import CNN
from data import MNIST

if __name__ == '__main__':
    
    learning_rate = 0.01
    training_epochs = 20
    batch_size = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, test_data, syn_data = MNIST()
    #syn_loader = MNIST(_type='syn', batch_size=batch_size)

    model = CNN().to(device)    

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X, Y = train_data
    X_test, Y_test = test_data
    X_syn, Y_syn = syn_data
    X = np.array_split(X, len(X) / batch_size)
    Y = np.array_split(Y, len(Y) / batch_size)
 
    X_syn = np.array_split(X_syn, len(X_syn) / batch_size)
    Y_syn = np.array_split(Y_syn, len(Y_syn) / batch_size)    
    labels = np.eye(10)

    for epoch in range(training_epochs):
        avg_loss = 0

        model.train()
        for x_syn, y_syn in zip(X_syn, Y_syn):
            x_syn = torch.Tensor(x_syn.reshape(batch_size, 1, 28, 28)).to(device)
            y_syn = torch.Tensor(labels[y_syn].reshape(batch_size, 10))

            optimizer.zero_grad()
            y_syn_pred = model(x_syn)
            loss = loss_fn(y_syn_pred, y_syn)
            loss.backward()
            optimizer.step()
            avg_loss += loss / len(X_syn)

        ## model 파라미터 진짜 안바뀌는지 확인 필요
        model.eval()
        for x_train_, y_train, x_syn in zip(X, Y, X_syn):
            x_train_ = torch.Tensor(x_train_.reshape(batch_size, 1, 28, 28))
            x_train = x_train_.detach().to(device)
            x_train.requires_grad = True
            y_train = torch.Tensor(labels[y_train].reshape(batch_size, 10))

            syn_opt = torch.optim.Adam([x_train], lr=learning_rate)

            syn_opt.zero_grad()
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            syn_opt.step()

            x_syn = x_syn - np.array(x_train.grad)
            x_train.detach()
        
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch, avg_loss))
            with open(f"./result/test_epoch_{epoch+1}.pkl","wb") as f:
                pickle.dump(X_syn[0], f)
    with open(f"./result/test_last.pkl","wb") as f:
        pickle.dump(X_syn, f)

    model.eval()
    accuracy = 0
    with torch.no_grad():
        x_test = torch.Tensor(X_test.reshape(len(X_test), 1, 28, 28)).to(device)
        y_test = torch.Tensor(Y_test)

        prediction = model(x_test)
        correct_prediction = torch.argmax(prediction, 1) == y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
    
    ## syn data 저장