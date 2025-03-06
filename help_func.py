import torch
import torch.nn.functional as F

from loss_func import custom_loss

def self_feeding(model, xuk, Num_meas):
    initial_input = xuk[:, 0, :]
    num_steps = int(len(xuk[0, :, 0]))
    inputs = xuk[:,:,Num_meas:]

    predictions = []
    predictions.append(initial_input)

    for step in range(num_steps - 1):
        x_pred = model(initial_input)
        x_pred = torch.cat((x_pred, inputs[:, step, :]), dim=1)
        predictions.append(x_pred.detach())
        initial_input = x_pred

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk)

    return predictions, loss


def enc_self_feeding(model, xuk, Num_meas):
    x_k = xuk[:, 0, :Num_meas]
    u = xuk[:, :, Num_meas:]

    num_steps = int(len(xuk[0, :, 0]))
    predictions = []
    predictions.append(x_k)

    y_k = model.x_Encoder(x_k)
    for m in range(0, num_steps-1):

        v = model.u_Encoder(torch.cat((x_k, u[:, m, :]), dim=1))
        y_k = model.x_Koopman_op(y_k) + model.u_Koopman_op(v)
        x_k = model.x_Decoder(y_k)
        predictions.append(x_k)

    predictions = torch.stack(predictions, dim=1)
    loss = custom_loss(predictions, xuk[:, :, :Num_meas])

    return predictions, loss
