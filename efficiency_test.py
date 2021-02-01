import argparse
import torch
import timm
import time
from models import cnn,lstm,mlp
from const import MODELS

def arg_parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model',
                        type=str,
                        default='cnn',
                        choices=['cnn','mlp','lstm','resnet50','mobilenet_v2','efficientnet_lite4'],
                        help='type of models')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64)

    parser.add_argument('--back_prop',
                        type=bool,
                        default=True)

    parser.add_argument('--logger',
                        type=bool,
                        default=False)
    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        choices=['sgd','adam'])
    parser.add_argument("--round",
                        type=int,
                        default=100)

    args = parser.parse_args()

    return args

def create_dummy_data(input_dim:set,output_dim:set):

    in_data = torch.rand(input_dim)
    out_data = torch.rand(output_dim)

    return in_data,out_data


if __name__ == "__main__":

    args = arg_parse()

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print("Using CUDA backend")

    else:
        device = "cpu"
        print("Using CPU")


    if args.model == "cnn":
        model = cnn.CNN()
        in_dim = (args.batch_size,3,24,24)
        out_dim = (args.batch_size,10)
    elif args.model == "mlp":
        model = mlp.Model()
        in_dim = (args.batch_size,1,28,28)
        out_dim = (args.batch_size,10)
    elif args.model == "lstm":
        model = lstm.Model()
        in_dim = (args.batch_size,80)
        out_dim = (args.batch_size,1)
    elif args.model == "mobilenet_v2":
        model = timm.create_model("mobilenetv2_100",pretrained=False)
        in_dim = (args.batch_size,3,224,224)
        out_dim = (args.batch_size,1000)
    elif args.model == "resnet50":
        model = timm.create_model("resnet50",pretrained=False)
        in_dim = (args.batch_size,3,224,224)
        out_dim = (args.batch_size,1000)
    else:
        model = timm.create_model("tf_efficientnet_lite4",pretrained=False)
        in_dim = (args.batch_size,3,224,224)
        out_dim = (args.batch_size,1000)

    model.to(device)
    in_data, loss = create_dummy_data(in_dim,out_dim)
    in_data, loss = in_data.to(device), loss.to(device)
    forward_cost, backward_cost = 0.0, 0.0
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

    for i in range(args.round):
        tic = time.time()
        out = model(in_data)
        toc = time.time()
        forward_cost += (toc - tic) * 1000

        if args.back_prop:
            tic = time.time()
            model.zero_grad()
            grad = out.backward(loss)
            optimizer.step()
            toc = time.time()
            backward_cost += (toc - tic) * 1000

        print(f"round: {i}| forward_cost: {forward_cost}| backward_cost: {backward_cost}")

    print(forward_cost/args.round)

    if args.back_prop:
        print(backward_cost/args.round)





