"""flower-tutorial: A Flower / PyTorch app."""

import math
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_tutorial.task import Net, load_data
from flower_tutorial.task import test as test_fn
from flower_tutorial.task import train as train_fn

from collections import Counter
import numpy as np

log_path = "all_clients_distribution.txt"

from flwr.common import (
    log,
)
from logging import INFO


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _, _ = load_data(partition_id, num_partitions)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.train("quality_measurement")
def custom_action(message: Message, context: Context) -> Message:
    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _, paramethers = load_data(partition_id, num_partitions)
    
    accuracy = 0.0
    completeness = 0.0
    class_balance = 0.0
 
    #total = len(trainloader.dataset)
    #completeness = 1 - (sum((e["img"] >= 1.0).all().item() for e in trainloader.dataset) / len(trainloader.dataset))
    #log(INFO, "Completeness_calculated: %s", completeness)
    completeness = 1 - paramethers[0]
    accuracy = 1 - paramethers[1]
    #log(INFO, "Completeness_sent: %s", completeness)
    
    labels = [e["label"] for e in trainloader.dataset]
    counts = Counter(labels)
    total = len(labels)
    fractions = [counts[i]/total for i in range(10)]
    class_balance = -sum(f*math.log(f) for f in fractions if f>0)/math.log(10)
    #log(INFO, "Class balance: %s", class_balance)
   
    
    metrics = {
        "accuracy": accuracy,
        "completeness": completeness,
        "class_balance": class_balance
       # "num-examples": len(trainloader.dataset),
    }
    
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=message)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader, _ = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)



