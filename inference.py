import torch
from model import Cnn14_DecisionLevelAtt
from config import labels, classes_num, model_path

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.
                      backends.mps.is_available() else 'cpu')


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


class EventDetector(object):

    def __init__(self):
        self.labels = labels
        self.classes_num = classes_num

        self.model = Cnn14_DecisionLevelAtt(sample_rate=32000,
                                            window_size=1024,
                                            hop_size=320,
                                            mel_bins=64,
                                            fmin=50,
                                            fmax=14000,
                                            classes_num=self.classes_num,
                                            interpolate_mode='nearest')

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(device)
        print('Using', device)

        # self.model = torch.compile(self.model) for python 3.10 or pytorch nightly

    def inference(self, audio):
        audio = move_data_to_device(audio, device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(input=audio)

        framewise_output = output_dict['framewise_output'].data.cpu().numpy()

        return framewise_output
