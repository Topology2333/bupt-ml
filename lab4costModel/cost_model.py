import torch
import torch.nn as nn

from plan import Operator

operators = ["Projection", "Selection", "Sort", "HashAgg", "HashJoin", "TableScan", "IndexScan", "TableReader",
             "IndexReader", "IndexLookUp"]


# There are many ways to extract features from plan:
# 1. The simplest way is to extract features from each node and sum them up. For example, we can get
#      a  the number of nodes;
#      a. the number of occurrences of each operator;
#      b. the sum of estRows for each operator.
#    However we lose the tree structure after extracting features.
# 2. The second way is to extract features from each node and concatenate them in the DFS traversal order.
#                  HashJoin_1
#                  /          \
#              IndexJoin_2   TableScan_6
#              /         \
#          IndexScan_3   IndexScan_4
#    For example, we can concatenate the node features of the above plan as follows:
#    [Feat(HashJoin_1)], [Feat(IndexJoin_2)], [Feat(IndexScan_3)], [Padding], [Feat(IndexScan_4)], [Padding], [Padding], [Feat(TableScan_6)], [Padding], [Padding]
#    Notice1: When we traverse all the children in DFS, we insert [Padding] as the end of the children. In this way, we
#    have an one-on-one mapping between the plan tree and the DFS order sequence.
#    Notice2: Since the different plans have the different number of nodes, we need padding to make the lengths of the
#    features of different plans equal.
class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: define variables to collect features from plans
        self.num_nodes = 0
        self.operator_counts = {op: 0 for op in operators}
        self.operator_est_rows = 0
        pass

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: extract features from op
        for u in operators:
            if(op.id.find(u) != -1):
                self.num_nodes += 1
                self.operator_counts[u] += 1
                self.operator_est_rows += float(op.est_rows)
        # pass

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        # YOUR CODE HERE: process and return the features
        features = [self.num_nodes]
        for op in operators:
            features.append(self.operator_counts[op])
        features.append(self.operator_est_rows)
        return features
        # pass



class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans, max_operator_num):
        super().__init__()
        self.data = []
        for plan in plans:
            collector = PlanFeatureCollector()
            vec = collector.walk_operator_tree(plan.root)
            # YOUR CODE HERE: maybe you need padding the features if you choose the second way to extract the features.
            # print(vec)
            features = torch.Tensor(vec)
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            # print(features)
            # print(exec_time)
            self.data.append((features, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Define the layers of your model
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        # Initialize the weights of your model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


def count_operator_num(op: Operator):
    num = 2  # one for the node and another for the end of children
    for child in op.children:
        num += count_operator_num(child)
    return num


def estimate_learning(train_plans, test_plans):
    max_operator_num = 0
    for plan in train_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    for plan in test_plans:
        max_operator_num = max(max_operator_num, count_operator_num(plan.root))
    print(f"max_operator_num:{max_operator_num}")

    train_dataset = PlanDataset(train_plans, max_operator_num)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=1)

    model = YourModel(input_size=max_operator_num)
    model.init_weights()

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        # Calculate the absolute differences between actual and estimated times
        abs_diff = torch.abs(act_time - est_time)
        # Sum the absolute differences
        loss = torch.sum(abs_diff)
        # Divide by 'n' (number of samples)
        loss /= len(act_time)
        return loss
        # pass

    # YOUR CODE HERE: complete training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 40
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        model.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            features, exec_time = data

            optimizer.zero_grad()
            estimate_exec_time = model(features)
            # print(estimate_exec_time) 无实意代码
            # print("gap")
            # print(exec_time)
            loss = loss_fn(estimate_exec_time, exec_time)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Average Loss: {average_loss}")
        # pass
    model.eval()
    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        features, exec_time = data
        estimated_exec_time = model(features).detach().numpy()  # Convert to NumPy array
        train_est_times.extend(estimated_exec_time)
        train_act_times.extend(exec_time.numpy())  # Convert to NumPy array
        # pass

    test_dataset = PlanDataset(test_plans, max_operator_num)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=1)

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        # YOUR CODE HERE: evaluate on test data
        features, exec_time = data
        outputs = model(features)
        test_est_times.extend(outputs.tolist())
        test_act_times.extend(exec_time.tolist())
        # pass
    
    return train_est_times, train_act_times, test_est_times, test_act_times