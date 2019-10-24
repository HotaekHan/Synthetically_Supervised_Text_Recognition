import torch.nn as nn
import torch
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    def __init__(self, num_In, num_Hidden, num_Out):
        super(BidirectionalLSTM, self).__init__()

        # structure of input of LSTM is (sequence length , batch, features)
        # structure of output of LSTM is (sequence length, batch, hidden)
        self.lstm = nn.LSTM(num_In, num_Hidden, bidirectional=True)
        # size of input is hidden * 2 because it is bidirectional
        self.embedding = nn.Linear(num_Hidden * 2, num_Out)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        seq_len, batch, hidden = lstm_out.size()
        lstm_out = lstm_out.view(seq_len * batch, hidden)

        out = self.embedding(lstm_out)
        out = out.view(seq_len, batch, -1)

        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        conv_maps = [64, 128, 256, 256, 512, 512, 512]

        self.conv1 = nn.Conv2d(3, conv_maps[0], kernel_size=3, stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(conv_maps[0], conv_maps[1], kernel_size=3, stride=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(conv_maps[1], conv_maps[2], kernel_size=3, stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(conv_maps[2], conv_maps[3], kernel_size=3, stride=(2, 1), padding=1)
        self.conv5 = nn.Conv2d(conv_maps[3], conv_maps[4], kernel_size=3, stride=(1, 1), padding=1, bias=False)
        self.conv5_bn = nn.BatchNorm2d(conv_maps[4])
        self.conv6 = nn.Conv2d(conv_maps[4], conv_maps[5], kernel_size=3, stride=(2, 1), padding=1, bias=False)
        self.conv6_bn = nn.BatchNorm2d(conv_maps[5])
        self.conv7 = nn.Conv2d(conv_maps[5], conv_maps[6], kernel_size=3, stride=(2, 1), padding=1)

    def forward(self, inputs):
        # input -> 3x32x100'
        out = self.conv1(inputs)
        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))
        out = self.conv3(F.leaky_relu(out, negative_slope=0.2))
        out = self.conv4(F.leaky_relu(out, negative_slope=0.2))
        out = self.conv5_bn(self.conv5(F.leaky_relu(out, negative_slope=0.2)))
        out = self.conv6_bn(self.conv6(F.leaky_relu(out, negative_slope=0.2)))
        out = self.conv7(F.leaky_relu(out, negative_slope=0.2))

        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        # Encoder
        conv_maps = [64, 128, 256, 256, 512, 512, 512]
        LSTM_hiddens = [256, 256]

        self.lstm1 = BidirectionalLSTM(num_In=conv_maps[6], num_Hidden=LSTM_hiddens[0], num_Out=LSTM_hiddens[0])
        self.lstm2 = BidirectionalLSTM(num_In=LSTM_hiddens[0], num_Hidden=LSTM_hiddens[1], num_Out=num_classes)

    def forward(self, inputs):
        batch, channel, height, width = inputs.size()
        assert height == 1, "the height of conv must be 1"
        inputs = inputs.squeeze(2)  # batch x channel x width
        inputs = inputs.permute(2, 0, 1)  # batch x channel x width -> width x batch x channel(26 x batch x 512)
        # maximum of time step is width(26 in case)
        # size of feature vector is channel(512 in case)

        out = self.lstm1(inputs)
        out = self.lstm2(out)

        return out


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes=num_classes)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._do_initializer()

    def _do_initializer(self):
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.normal_(tensor=module.weight, mean=0, std=0.01)

        for weights in self.decoder.lstm1.lstm.all_weights:
            for weight in weights:
                if len(weight.shape) > 1:
                    # nn.init.orthogonal_(weight, gain=1)
                    nn.init.normal_(tensor=weight, mean=0, std=0.01)
        nn.init.normal_(tensor=self.decoder.lstm1.embedding.weight, mean=0, std=0.01)

        for weights in self.decoder.lstm2.lstm.all_weights:
            for weight in weights:
                if len(weight.shape) > 1:
                    # nn.init.orthogonal_(weight, gain=1)
                    nn.init.normal_(tensor=weight, mean=0, std=0.01)
        nn.init.normal_(tensor=self.decoder.lstm2.embedding.weight, mean=0, std=0.01)

    def forward(self, inputs):
        # input -> 1x32x100
        out = self.encoder(inputs)
        out = self.decoder(out)
        out = self.log_softmax(out)

        return out

def load_model(num_classes):
    net = CRNN(num_classes=num_classes)

    return net

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(num_classes=37).to(device)

    num_parameters = 0.
    for param in net.parameters():
        sizes = param.size()

        num_layer_param = 1.
        for size in sizes:
            num_layer_param *= size
        num_parameters += num_layer_param

    print(net)
    print("num. of parameters : " + str(num_parameters))

    inputs = torch.FloatTensor(2, 3, 32, 100).to(device)
    target_lengths = torch.IntTensor([9, 11]).to(device)
    targets = torch.LongTensor([29, 22, 19, 24, 21, 19, 24, 17, 0, 28, 15, 23, 25, 14, 15, 22, 15, 28, 29, 0]).to(device)

    # with torch.set_grad_enabled(False):
    #     preds = net(inputs, is_training=False, target_lengths=None, targets=None)
    #     print(preds.size())
    #     print(inputs.requires_grad)
    #     print(preds.requires_grad)

    with torch.set_grad_enabled(True):
        preds = net(inputs)
        print(preds.size())
        print(inputs.requires_grad)
        print(preds.requires_grad)

if __name__ == '__main__':
    test()