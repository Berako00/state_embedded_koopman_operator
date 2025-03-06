import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def debug_L12(xuk, encoder, decoder):
    actual = torch.zeros(xuk.shape[0], len(xuk[0, :, 0]),xuk.shape[2], dtype=torch.float32)
    prediction = torch.zeros(xuk.shape[0], len(xuk[0, :, 0]),xuk.shape[2], dtype=torch.float32)

    for m in range(0,len(xuk[0, :, 0])):
        prediction[:, m, :] = decoder(encoder(xuk[:, m, :]))
        actual[:, m, :]  = xuk[:, m, :]

    return actual, prediction

def debug_L3(xuk, Num_meas, model):
    prediction = model.x_Decoder(model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas])) + model.u_Koopman_op(model.u_Encoder(xuk[:, 0, :])))
    actual = xuk[:, 1, :Num_meas]

    return actual, prediction

def debug_L4(xuk, Num_meas, model):
    prediction = model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas])) + model.u_Koopman_op(model.u_Encoder(xuk[:, 0, :]))
    actual = model.x_Encoder(xuk[:, 1, :Num_meas])

    return actual, prediction

def debug_L5(xuk, Num_meas, S_p, model):
    u = xuk[:, :, Num_meas:]
    prediction = torch.zeros(xuk.shape[0], S_p+1, Num_meas, dtype=torch.float32)
    actual = xuk[:, :(S_p + 1),:]
    prediction[:, 0, :] = xuk[:, 0, :Num_meas]
    x_k = model.x_Decoder(model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas])) + model.u_Koopman_op(model.u_Encoder(xuk[:, 0, :])))
    prediction[:, 1, :] = x_k

    for m in range(1, S_p):
        xukh = torch.cat((x_k, u[:, m, :]), dim=1)
        x_k  = model.x_Decoder(model.x_Koopman_op(model.x_Encoder(x_k)) + model.u_Koopman_op(model.u_Encoder(xukh)))
        prediction[:, m+1, :] = x_k

    return actual, prediction

def debug_L6(xuk, Num_meas, Num_x_Obsv, T, model):
    prediction = torch.zeros(xuk.shape[0], T, Num_x_Obsv, dtype=torch.float32)
    actual = torch.zeros(xuk.shape[0], T, Num_x_Obsv, dtype=torch.float32)

    actual[:, 0 ,:] = model.x_Encoder(xuk[:, 0, :Num_meas])
    actual[:, T-1 ,:] = model.x_Encoder(xuk[:, T-1, :Num_meas])

    u = xuk[:, :, Num_meas:]
    prediction[:, 0, :] = model.x_Encoder(xuk[:, 0, :Num_meas])
    y_k = model.x_Koopman_op(model.x_Encoder(xuk[:, 0, :Num_meas])) + model.u_Koopman_op(model.u_Encoder(xuk[:, 0, :]))
    prediction[:, 1, :] = y_k
    x_k = model.x_Decoder(y_k)

    for m in range(1, T-1):
        actual[:, m ,:] = model.x_Encoder(xuk[:, m, :Num_meas])
        v = model.u_Encoder(torch.cat((x_k, u[:, m, :]), dim=1))
        y_k = model.x_Koopman_op(y_k) + model.u_Koopman_op(v)
        x_k = model.x_Decoder(y_k)
        prediction[:, m+1, :] = y_k

    return actual, prediction
