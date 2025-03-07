import torch
import torch.nn as nn
import torch.nn.functional as F

class AUTOENCODER(nn.Module):
    def __init__(self, Num_meas, Num_inputs, Num_x_Obsv, Num_x_Neurons, Num_u_Obsv, Num_u_Neurons, Num_hidden_x_encoder, Num_hidden_x_decoder, Num_hidden_u_encoder, Num_hidden_u_decoder):

        super(AUTOENCODER, self).__init__()
        self.num_meas = Num_meas

        self.x_Encoder_In = nn.Linear(Num_meas, Num_x_Neurons)
        self.x_encoder_hidden = nn.ModuleList([nn.Linear(Num_x_Neurons, Num_x_Neurons) for _ in range(Num_hidden_x_encoder)])
        self.x_Encoder_out = nn.Linear(Num_x_Neurons, Num_x_Obsv)

        self.x_Koopman = nn.Linear(Num_x_Obsv, Num_x_Obsv, bias=False)

        self.x_Decoder_In = nn.Linear(Num_x_Obsv, Num_x_Neurons)
        self.x_decoder_hidden = nn.ModuleList([nn.Linear(Num_x_Neurons, Num_x_Neurons) for _ in range(Num_hidden_x_decoder)])
        self.x_Decoder_out = nn.Linear(Num_x_Neurons, Num_meas)

        self.u_Encoder_In = nn.Linear(Num_meas + Num_inputs, Num_u_Neurons)
        self.u_encoder_hidden = nn.ModuleList([nn.Linear(Num_u_Neurons, Num_u_Neurons) for _ in range(Num_hidden_u_encoder)])
        self.u_Encoder_out = nn.Linear(Num_u_Neurons, Num_u_Obsv)

        self.u_Koopman = nn.Linear(Num_u_Obsv, Num_x_Obsv, bias=False)

        self.u_Decoder_In = nn.Linear(Num_u_Obsv, Num_u_Neurons)
        self.u_decoder_hidden = nn.ModuleList([nn.Linear(Num_u_Neurons, Num_u_Neurons) for _ in range(Num_hidden_u_decoder)])
        self.u_Decoder_out = nn.Linear(Num_u_Neurons, Num_meas + Num_inputs)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)*4
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def x_Encoder(self, x):
        x = x[:, :self.num_meas]
        x = F.relu(self.x_Encoder_In(x))
        for layer in self.x_encoder_hidden:
            x = F.relu(layer(x))
        x = self.x_Encoder_out(x)
        return x

    def x_Koopman_op(self, x):
        return self.x_Koopman(x)

    def x_Decoder(self, x):
        x = F.relu(self.x_Decoder_In(x))
        for layer in self.x_decoder_hidden:
            x = F.relu(layer(x))
        x = self.x_Decoder_out(x)
        return x

    def u_Encoder(self, x):
        x = F.relu(self.u_Encoder_In(x))
        for layer in self.u_encoder_hidden:
            x = F.relu(layer(x))
        x = self.u_Encoder_out(x)
        return x

    def u_Koopman_op(self, x):
        return self.u_Koopman(x)

    def u_Decoder(self, x):
        x = F.relu(self.u_Decoder_In(x))
        for layer in self.u_decoder_hidden:
            x = F.relu(layer(x))
        x = self.u_Decoder_out(x)
        return x

    def forward(self, x_k):
        y_k = self.x_Encoder(x_k)
        v_k = self.u_Encoder(x_k)
        y_k1 = self.x_Koopman_op(y_k) + self.u_Koopman_op(v_k)
        x_k1 = self.x_Decoder(y_k1)
        return x_k1
